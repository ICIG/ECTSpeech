import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from model.como import GradLogPEstimator2d, BaseModule

class ComoECT(BaseModule):
    def __init__(self):
        super(ComoECT, self).__init__()
        self.denoise_fn = GradLogPEstimator2d(64, n_feats=80)
        
        # 创建EMA版本的模型，仅用于推理
        self.denoise_fn_ema = copy.deepcopy(self.denoise_fn)
        # 禁用EMA模型的梯度
        for param in self.denoise_fn_ema.parameters():
            param.requires_grad = False
        
        # ECT微调参数
        self.q = 1.5  # 衰减因子
        
        self.n_mels = 80  # 梅尔频带数
        self.P_mean = -1.2  # 噪声分布参数 (更新为-1.2)
        self.P_std = 1.2  # 噪声分布参数 (更新为1.2)
        self.sigma_data = 0.5  # sigma_data
        
        self.sigma_min = 0.002  # 最小噪声水平
        self.sigma_max = 80  # 最大噪声水平
        self.rho = 7  # EDM噪声调度参数
        
        self.N = 50  # 时间步数
        
        # 数值稳定性参数
        self.eps = 1e-6  # 防止除零
        self.ratio_clamp = True  # 是否对比率进行裁剪
        
        # 损失缩放
        self.loss_scale = 1.0
        
        # EMA衰减率 - 更新为与ect-main保持一致
        self.ema_decay = 0.9999  # 基础EMA衰减率
        self.ema_halflife_nimg = 10000  # 半衰期图像数量
        self.ema_rampup_ratio = 0.05  # EMA预热比例
        
        # 自适应权重参数
        self.adap_c = 1e-3  # 自适应权重平滑因子

        # 映射函数参数
        self.k = 8.0  # 映射函数k参数
        self.b = 1.0  # 映射函数b参数
        
        # 迭代计数相关
        self.iter_d = 2000  # 迭代步长
        self.iter_count = 0  # 全局迭代计数器
        
        # 预计算时间步
        self.eps_tensor = {}  # 设备缓存字典
        step_indices = torch.arange(self.N)   
        t_steps = (self.sigma_min ** (1 / self.rho) + step_indices / (self.N - 1) * (self.sigma_max ** (1 / self.rho) - self.sigma_min ** (1 / self.rho))) ** self.rho
        #self.t_steps = torch.cat([torch.zeros_like(t_steps[:1]), torch.as_tensor(t_steps)])
        self.t_steps = torch.cat([torch.zeros_like(t_steps[:1]), self.round_sigma(t_steps)])  
         
        # 初始化损失函数
        self.loss_fn = ECMLoss(
            P_mean=self.P_mean, 
            P_std=self.P_std, 
            sigma_data=self.sigma_data,
            q=self.q, 
            c=self.adap_c, 
            k=self.k, 
            b=self.b,
            loss_scale=self.loss_scale
        )
        
        print(f"ComoECT 模型初始化: stage={self.stage}, ratio={self.ratio:.4f}, loss_scale={self.loss_scale}")

    # 添加属性getter，保持与现有代码兼容
    @property
    def stage(self):
        """获取当前ECT阶段，从loss_fn中读取"""
        return self.loss_fn.stage
        
    @property
    def ratio(self):
        """获取当前ECT比例，从loss_fn中读取"""
        return self.loss_fn.ratio

    def update_schedule(self, stage=None):
        """更新ECT微调阶段，与原始ECT项目保持一致
        
        阶段计算基于迭代计数或直接提供的stage值
        每个阶段对应不同的ratio，用于控制t到r的映射关系
        
        Args:
            stage: 新的微调阶段，如果为None则使用迭代计数器自动计算
        """
        # 如果没有提供stage，则根据iter_count自动计算
        if stage is None:
            stage = self.iter_count // self.iter_d
        
        # 保存旧值用于日志
        old_ratio = self.ratio
        
        # 直接更新loss_fn的状态
        self.loss_fn.update_schedule(stage)
        
        # 重置设备缓存
        self.eps_tensor = {}  
        
        # 输出日志
        print(f"更新ECT微调阶段: {stage}, 比例: {self.ratio:.4f} (之前: {old_ratio:.4f})")
        
        if stage == 0:
            print("- 当前是初始阶段，r接近0，相当于纯扩散模型训练")
        elif stage == 1:
            print("- 进入第一阶段一致性训练，以温和的方式开始ECT调优")
            print(f"- 映射公式: r = t * {self.ratio:.4f}")
        else:
            print(f"- 进入深度一致性阶段({stage})，强化ECT一致性映射")
            print(f"- 映射公式: r = t * {self.ratio:.4f}")
            
        # 打印当前q值
        print(f"- 使用q值: {self.q}, 当前decay: {1/self.q**stage:.4f}")
        
        # 输出一些采样值示例，帮助理解t和r的关系
        sample_ts = [0.1, 1.0, 10.0, self.sigma_max]
        print("- t到r的映射示例:")
        for t in sample_ts:
            r = t * self.ratio
            print(f"  t={t:.2f} -> r={r:.2f}")
            
        # 阶段0特别处理
        if stage == 0:
            print("- 注意: 阶段0中r始终为0，这是标准扩散模型训练阶段")
            
        # 阶段1+说明
        else:
            print(f"- 阶段{stage}中r随t增长，但增长率为原来的{self.ratio*100:.1f}%")
            print("- 这种映射确保了从t->r的平滑过渡，便于EDM到ECT的渐进式微调")

    def update_ema(self):
        """更新EMA模型权重 - 修改为与ect-main一致的实现
        
        注意：应在每个训练批次后调用此函数
        """
        with torch.no_grad():
            # 计算当前EMA衰减率
            batch_size = 16  # 可以根据实际情况调整
            ema_nimg = self.ema_halflife_nimg
            if self.ema_rampup_ratio is not None:
                ema_nimg = min(ema_nimg, self.iter_count * self.ema_rampup_ratio)
            ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))
            
            # 应用EMA更新
            for p_ema, p_net in zip(self.denoise_fn_ema.parameters(), self.denoise_fn.parameters()):
                p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))
        
        # 增加迭代计数
        self.iter_count += 1

    def load_teacher_weights(self, state_dict, verbose=False):
        """从教师模型加载权重
        
        Args:
            state_dict: 教师模型的状态字典
            verbose: 是否输出详细信息
        """
        try:
            # 检查状态字典中的键的数量
            if verbose:
                print(f"状态字典包含 {len(state_dict)} 个键")
                
                # 打印一些状态字典中的键示例
                sample_keys = list(state_dict.keys())[:5]
                print(f"键示例: {sample_keys}")
                
                # 检查当前模型的参数数量
                current_param_count = sum(1 for _ in self.denoise_fn.parameters())
                print(f"当前模型的denoise_fn参数数量: {current_param_count}")
                
            # 保存加载前的一些参数统计，用于对比
            before_stats = {}
            for name, param in self.denoise_fn.named_parameters():
                before_stats[name] = param.data.mean().item()
            
            # 尝试加载去噪网络权重
            missing, unexpected = self.denoise_fn.load_state_dict(state_dict, strict=False)
            
            # 检查参数是否发生变化
            after_stats = {}
            changed_params = 0
            for name, param in self.denoise_fn.named_parameters():
                after_stats[name] = param.data.mean().item()
                if name in before_stats and abs(after_stats[name] - before_stats[name]) > 1e-6:
                    changed_params += 1
            
            if verbose:
                print(f"参数变化检测: 共有 {changed_params}/{len(before_stats)} 个参数发生变化")
                
                if changed_params == 0:
                    print("警告: 没有参数发生变化，可能加载失败")
            
            # 同步更新EMA模型
            self.denoise_fn_ema = copy.deepcopy(self.denoise_fn)
            for param in self.denoise_fn_ema.parameters():
                param.requires_grad = False
            
            if verbose:
                print("已创建EMA模型的新副本")
                
                if len(missing) > 0:
                    print(f"缺少的参数: {len(missing)} 个")
                    if len(missing) < 20:
                        print(f"缺少的参数: {missing}")
                if len(unexpected) > 0:
                    print(f"未预期的参数: {len(unexpected)} 个")
                    if len(unexpected) < 20:
                        print(f"未预期的参数: {unexpected}")
            
            # 重置ECT微调状态
            self.iter_count = 0
            
            # 更新loss_fn的状态
            self.loss_fn.stage = 0
            self.loss_fn.ratio = 0.0
            
            if verbose:
                print(f"重置ECT微调状态: stage={self.stage}, ratio={self.ratio:.4f}")
            
            return True
        except Exception as e:
            print(f"加载教师模型权重失败: {str(e)}")
            return False

    def t_to_r(self, t):
        
        return self.loss_fn.t_to_r(t)

    def round_sigma(self, sigma):
        
        return torch.as_tensor(sigma)

    def get_sampling_steps(self, n_steps):
        
        if n_steps == 1:
            return torch.tensor([self.t_steps[-1]])
        
        # 计算gamma因子
        min_inv_rho = self.sigma_min ** (1 / self.rho)
        max_inv_rho = self.sigma_max ** (1 / self.rho)
        
        # 创建均匀间隔的索引
        step_indices = torch.arange(n_steps).to(torch.float32) / (n_steps - 1)
        
        # 计算对应的噪声水平
        t = (min_inv_rho + step_indices * (max_inv_rho - min_inv_rho)) ** self.rho
        
        # 按降序返回，确保从高噪声水平开始
        return torch.flip(t, [0])

    def EDMPrecond(self, x, sigma, cond, denoise_fn, nonpadding):
       
        # 检查掩码的有效性
        if nonpadding.sum() < 1:
            # 返回零张量，但保留梯度连接
            return torch.zeros_like(x, requires_grad=True)
        
        # 确保sigma是适当的形状
        sigma = sigma.reshape(-1, 1, 1)
        
        # 保持与como.py完全一致的实现
        c_skip = self.sigma_data ** 2 / ((sigma-self.sigma_min) ** 2 + self.sigma_data ** 2)
        c_out = (sigma-self.sigma_min) * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4
        
        # 应用去噪模型，与como.py保持完全一致的调用方式
        F_x = denoise_fn((c_in * x), nonpadding, cond, c_noise.flatten())
        D_x = c_skip * x + c_out * F_x
        
        return D_x

    def forward(self, x, nonpadding, cond, t_steps=1, infer=False, use_ema=True, use_ct_sampler=True, use_ect_generator=True):
        """前向传播函数，支持训练和推理
        
        Args:
            x: 输入特征
            nonpadding: 掩码
            cond: 条件信息
            t_steps: 采样步数（推理时使用）
            infer: 是否为推理模式
            use_ema: 是否使用EMA模型（推理时使用）
            use_ct_sampler: 是否使用CT采样器（已废弃，保留为了兼容性）
            use_ect_generator: 是否使用ECT_generator（优先级高于use_ct_sampler）
            
        Returns:
            训练模式: 损失值
            推理模式: 生成的梅尔谱图
        """
        if not infer:
            # 训练模式 - 计算ECT损失，使用主网络和EMA网络
            loss = self.loss_fn(x, nonpadding, cond, self.denoise_fn, self.denoise_fn_ema, self)
            # 应用损失缩放 - 从loss_fn移出到这里
            return loss * self.loss_scale
        else:
            # 推理模式 - 使用EMA网络生成梅尔谱图
            shape = (cond.shape[0], 80, cond.shape[2])
            x = torch.randn(shape, device=x.device)
            
            # 选择采样器
            if use_ect_generator:
                # 使用ECT_generator采样器 - 直接传入步数
                x = self.ECT_generator(x, cond, nonpadding, num_steps=t_steps, use_ema=use_ema)
            else:
                # 使用传统的CT_sampler
                x = self.CT_sampler(x, cond, nonpadding, t_steps, use_ema)
                
            return x

    def get_t_steps(self, N):
        N = N + 1  # 增加一步以匹配Como.py的实现
        step_indices = torch.arange(N)
        t_steps = (self.sigma_min ** (1 / self.rho) + step_indices / (N - 1) * (self.sigma_max ** (1 / self.rho) - self.sigma_min ** (1 / self.rho))) ** self.rho
        
        # 降序返回，确保从高噪声水平开始
        return t_steps.flip(0)
        
    def CT_sampler(self, latents, cond, nonpadding, t_steps=1, use_ema=True):        
        # 确定采样步骤
        if t_steps == 1:
            t_steps = [80]
        else:
            t_steps = self.get_t_steps(t_steps)
        
        # 转换为张量并移动到正确设备
        t_steps = torch.as_tensor(t_steps).to(latents.device)
        
        # 选择去噪网络
        denoise_fn = self.denoise_fn_ema if use_ema else self.denoise_fn
        
        # 初始化采样 - 使用纯噪声
        latents = latents * t_steps[0]
        latents = latents * nonpadding
        
        if len(t_steps) == 1:
            original_sigma = t_steps[0]  # 80
            enhanced_sigma = original_sigma #* 1.05  # 80 * 1.05 = 84
            x_raw = self.EDMPrecond(latents, enhanced_sigma, cond, denoise_fn, nonpadding)
            temperature = 1
            x = x_raw * temperature
            x = x * nonpadding
        else:
            # 多步采样过程
            x = self.EDMPrecond(latents, t_steps[0], cond, denoise_fn, nonpadding)
            for t in t_steps[1:-1]:
                z = torch.randn_like(x)
                x_tn = x +  (t ** 2 - self.sigma_min ** 2).sqrt()*z
                x = self.EDMPrecond(x_tn, t, cond, denoise_fn, nonpadding)
        return x
        
    def ECT_generator(self, latents, cond, nonpadding, num_steps=1, use_ema=True, 
                      sigma_min=None, sigma_max=None, rho=None,
                      S_churn=0, S_min=0, S_max=float('inf'), S_noise=1):
        """
        ECT生成器，基于como.py的edm_sampler实现
        
        Args:
            latents: 初始噪声
            cond: 条件信息
            nonpadding: 掩码
            num_steps: 采样步数
            use_ema: 是否使用EMA模型
            sigma_min, sigma_max, rho: EDM参数，默认使用模型配置
            S_churn, S_min, S_max, S_noise: EDM采样器的随机化参数
        """
        
        # 选择使用哪个模型
        denoise_fn = self.denoise_fn_ema if use_ema else self.denoise_fn
        
        # 设置sigma参数，使用模型默认值
        sigma_min = sigma_min if sigma_min is not None else self.sigma_min  # 0.002
        sigma_max = sigma_max if sigma_max is not None else self.sigma_max  # 80
        rho = rho if rho is not None else self.rho                          # 7
        
        # Time step discretization (from como.py edm_sampler)
        num_steps = num_steps + 1
        step_indices = torch.arange(num_steps, device=latents.device)
        t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        t_steps = torch.cat([self.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])
        
        # Main sampling loop (from como.py edm_sampler)
        x_next = latents * t_steps[0]
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = x_next
            
            # Increase noise temporarily
            gamma = min(S_churn / (num_steps-1), np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
            t_hat = self.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur)
            
            # Euler step
            denoised = self.EDMPrecond(x_hat, t_hat, cond, denoise_fn, nonpadding)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur
            
            # Optional: Heun's 2nd order method (commented out in original)
            # if i < num_steps - 2:
            #     denoised = self.EDMPrecond(x_next, t_next, cond, denoise_fn, nonpadding)
            #     d_prime = (x_next - denoised) / t_next
            #     x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
        
        return x_next


class ECMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5, q=1.5, c=1e-3, k=8.0, b=1.0, adj='sigmoid', loss_scale=1.0):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        
        if adj == 'const':
            self.t_to_r = self.t_to_r_const
        elif adj == 'sigmoid':
            self.t_to_r = self.t_to_r_sigmoid
        else:
            raise ValueError(f'Unknow schedule type {adj}!')

        self.q = q
        self.stage = 0
        self.ratio = 0.
        
        self.k = k
        self.b = b
        self.c = c
        
        # 控制loss稳定性的附加参数
        self.min_tr_gap = 0.05     # 限制最小的 t - r 差距，防止被除得太大
        self.eps = 1e-8  # 防止除零
        print(f'P_mean: {self.P_mean}, P_std: {self.P_std}, q: {self.q}, k {self.k}, b {self.b}, c: {self.c}')

    def update_schedule(self, stage):
        self.stage = stage
        if stage == 0:
            self.ratio = 0.0
        else:
            self.ratio = 1 - 1 / self.q ** stage  # 移除+1，更符合数学逻辑

    def t_to_r_const(self, t):
        # 确保stage>0时使用正确的decay计算
        if self.stage == 0:
            return torch.zeros_like(t)
        else:
            decay = 1 / self.q ** self.stage  # 移除+1，更符合数学逻辑
            ratio = 1 - decay
            r = t * ratio
            return torch.clamp(r, min=0)

    def t_to_r_sigmoid(self, t):
        if self.stage == 0:
            r = torch.zeros_like(t)
        else:
            adj = 1 + self.k * torch.sigmoid(-self.b * t)
            decay = 1 / self.q ** self.stage  # 移除+1，更符合数学逻辑
            ratio = 1 - decay * adj
            r = t * ratio
            # --- 保护r的范围 ---
            if hasattr(self, 'ratio_clamp') and self.ratio_clamp:
                eps_tensor = torch.ones_like(t) * self.eps
                max_tensor = t * 0.99
                r = torch.max(eps_tensor, torch.min(r, max_tensor))
            else:
                r = torch.clamp(r, min=self.eps)
        return r

    def __call__(self, x_start, nonpadding, cond, denoise_fn, denoise_fn_ema,model):
        """计算ECT损失函数，适用于语音合成任务
        
        Args:
            x_start: 输入梅尔谱（目标）
            nonpadding: 掩码（指定有效区域）
            cond: 条件信息（即mu_y）
            denoise_fn: 主网络
            model: ComoECT模型实例，用于访问EDMPrecond等方法
        
        Returns:
            loss: 损失值张量，不包含缩放
        """
        # 修复：如果掩码全为零，返回与输入连接的零损失
        if nonpadding.sum() < 1:
            return torch.tensor(0.0, device=x_start.device, requires_grad=True)
            
        # t ~ p(t) 和 r ~ p(r|t, iters) (映射函数)
        rnd_normal = torch.randn([x_start.shape[0], 1, 1], device=x_start.device)
        t = (rnd_normal * self.P_std + self.P_mean).exp()
        t = torch.clamp(t, min=self.eps, max=model.sigma_max)
        r = self.t_to_r(t)

        # 没有augmentation，直接使用输入
        y = x_start
        
        # 共享噪声方向 - 不在噪声添加阶段应用掩码
        eps = torch.randn_like(y) 
        y_t = y + eps * t
        y_r = y + eps * r

        # 保存随机状态
        rng_state = torch.cuda.get_rng_state()
        
        D_yt = model.EDMPrecond(y_t, t, cond, denoise_fn, nonpadding)

        if r.max() > 0:
            torch.cuda.set_rng_state(rng_state)
            with torch.no_grad():
                D_yr = model.EDMPrecond(y_r, r, cond, denoise_fn, nonpadding)
            
            # 处理r=0的情况
            mask = (r > 0).reshape(-1, 1, 1)
            D_yr = torch.nan_to_num(D_yr)
            D_yr = mask * D_yr + (~mask) * y
        else:
            D_yr = y

        # 只在最终损失计算时应用掩码 - 与Como.py中CTLoss_D一致
        loss_squared = ((D_yt - D_yr) ** 2)
        loss_squared = loss_squared  * nonpadding

        # 计算每个样本的有效区域数量并归一化
        nonpadding_counts = nonpadding.reshape(nonpadding.shape[0], -1).sum(dim=1, keepdim=True) + 1e-8
        # 对每个样本进行归一化，考虑有效区域数量
        loss_per_sample = torch.sum(loss_squared.reshape(loss_squared.shape[0], -1), dim=-1) / nonpadding_counts.squeeze()

        # 自适应权重计算
        if self.c > 0:
            loss_per_sample = torch.sqrt(loss_per_sample + self.c ** 2) - self.c
        else:
            loss_per_sample = torch.sqrt(loss_per_sample)

        loss = loss_per_sample.mean()
        return loss