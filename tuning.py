import warnings
warnings.filterwarnings('ignore')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
from tqdm import tqdm
import copy
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import params
from model.tts import Comospeech
from data import TextMelDataset, TextMelBatchCollate
from utils import plot_tensor, save_plot, load_teacher_model, load_checkpoint
from text.symbols import symbols
from model.model_utils import check_tensor, set_debug_level, debug_print

DEBUG_VERBOSE = False  # 初始调试级别

train_filelist_path = params.train_filelist_path
valid_filelist_path = params.test_filelist_path
validation_filelist_path = params.valid_filelist_path  # 使用正确的验证集
cmudict_path = params.cmudict_path
add_blank = params.add_blank

log_dir = params.log_dir
n_epochs = params.n_epochs
batch_size = params.batch_size
out_size = params.out_size
learning_rate = params.learning_rate
random_seed = params.seed

nsymbols = len(symbols) + 1 if add_blank else len(symbols)
n_enc_channels = params.n_enc_channels
filter_channels = params.filter_channels
filter_channels_dp = params.filter_channels_dp
n_enc_layers = params.n_enc_layers
enc_kernel = params.enc_kernel
enc_dropout = params.enc_dropout
n_heads = params.n_heads
window_size = params.window_size

n_feats = params.n_feats
n_fft = params.n_fft
sample_rate = params.sample_rate
hop_length = params.hop_length
win_length = params.win_length
f_min = params.f_min
f_max = params.f_max
 
# ECT微调参数
double_every = params.double_every if hasattr(params, 'double_every') else 2  # 默认每2个epoch更新一次stage
max_grad_norm = getattr(params, 'grad_clip_threshold', 1.0)  # 梯度裁剪阈值
min_lr = getattr(params, 'ect_min_lr', 1e-6)  # 最小学习率，避免过度降低
loss_scale = getattr(params, 'loss_scale', 1.0)  # 损失缩放因子

# 迭代计数相关
iter_d = getattr(params, 'ect_iter_d', 2000)  # 迭代步长
start_iter = getattr(params, 'ect_start_iter', 0)  # 初始迭代数

# 学习率调整策略
def adjust_learning_rate(optimizer, lr_factor=0.5, min_lr=1e-6):
    """根据指定因子降低学习率，但确保不低于最小学习率"""
    current_lr = optimizer.param_groups[0]['lr']
    new_lr = max(current_lr * lr_factor, min_lr)
    
    if new_lr < current_lr:
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        print(f"学习率调整: {current_lr:.6f} -> {new_lr:.6f}")
    else:
        print(f"学习率已达最小值: {current_lr:.6f}，不再降低")
    
    return new_lr

# 修改save_model函数，只保存推理所需信息
def save_model(model, optimizer, iter_count, path, epoch=None):
    """保存模型到指定路径，只保存推理所需的信息
    
    Args:
        model: 要保存的模型
        optimizer: 优化器 (不再保存)
        iter_count: 当前迭代数 (不再保存)
        path: 保存路径
        epoch: 当前轮次 (不再保存)
    """
    # 只保存模型参数和ECT状态
    checkpoint = {
        'model': model.state_dict(),
        'stage': model.decoder.stage,
        'ratio': model.decoder.ratio
    }
    torch.save(checkpoint, path)
    
    print(f"已保存模型 (stage={model.decoder.stage}, ratio={model.decoder.ratio:.4f}): {path}")

# 新增：验证集损失计算函数（仅计算diffusion loss）
def calculate_validation_loss(model, validation_loader, out_size):
    """
    计算验证集扩散损失，不影响梯度
    
    Args:
        model: 训练中的模型
        validation_loader: 验证集数据加载器  
        out_size: 输出尺寸
        
    Returns:
        tuple: (avg_diff_loss, valid_batch_count)
    """
    model.eval()
    val_diff_losses = []
    valid_batch_count = 0
    
    with torch.no_grad():  # 不计算梯度
        for val_batch_idx, val_batch in enumerate(validation_loader):
            try:
                val_x, val_x_lengths = val_batch['x'].cuda(), val_batch['x_lengths'].cuda()
                val_y, val_y_lengths = val_batch['y'].cuda(), val_batch['y_lengths'].cuda()
                
                # 检查输入数据是否有效
                if not (check_tensor(val_x, "val_x") and check_tensor(val_y, "val_y")):
                    continue
                
                # 计算验证损失（只需要diffusion loss）
                _, _, val_diff_loss = model.compute_loss(
                    val_x, val_x_lengths, val_y, val_y_lengths, out_size=out_size)
                
                # 检查扩散损失是否有效
                if check_tensor(val_diff_loss, "val_diff_loss"):
                    val_diff_losses.append(val_diff_loss.item())
                    valid_batch_count += 1
                    
            except Exception as e:
                print(f"验证批次 {val_batch_idx} 计算时出错: {str(e)}")
                continue
    
    # 计算平均扩散损失
    if valid_batch_count > 0:
        avg_val_diff = np.mean(val_diff_losses)
        return avg_val_diff, valid_batch_count
    else:
        return None, 0

if __name__ == "__main__":
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    print("使用单GPU训练")

    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)
    print('初始化日志目录...')
    logger = SummaryWriter(log_dir=log_dir)

    # 初始化数据加载器
    print('初始化数据加载器...')
    train_dataset = TextMelDataset(train_filelist_path, cmudict_path, add_blank,
                                   n_fft, n_feats, sample_rate, hop_length,
                                   win_length, f_min, f_max)

    shuffle = True

    batch_collate = TextMelBatchCollate()
    loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                       collate_fn=batch_collate, drop_last=True, pin_memory=True, prefetch_factor=8,
                       num_workers=16, shuffle=shuffle, persistent_workers=True)
    
    # 估计每个epoch的迭代次数，用于日志记录
    steps_per_epoch = len(loader)
    print(f"每个epoch约有 {steps_per_epoch} 个批次")
    
    # 根据数据集大小调整iter_d参数（如果未手动设置）
    if not hasattr(params, 'ect_iter_d'):
        iter_d = steps_per_epoch * 2  # 大约2个epoch一个阶段
        print(f"自动设置iter_d={iter_d}（约2个epoch一个阶段）")
    
    test_dataset = TextMelDataset(valid_filelist_path, cmudict_path, add_blank,
                                  n_fft, n_feats, sample_rate, hop_length,
                                  win_length, f_min, f_max)

    #新增：真正的验证集数据加载器
    print('初始化验证集数据加载器...')
    validation_dataset = TextMelDataset(validation_filelist_path, cmudict_path, add_blank,
                                       n_fft, n_feats, sample_rate, hop_length,
                                       win_length, f_min, f_max)
    
    validation_batch_collate = TextMelBatchCollate()
    validation_loader = DataLoader(dataset=validation_dataset, batch_size=batch_size,
                                  collate_fn=validation_batch_collate, drop_last=False, 
                                  pin_memory=True, num_workers=8, shuffle=False)
    
    print(f'验证集包含 {len(validation_dataset)} 个样本，{len(validation_loader)} 个批次')

    print('初始化模型...')
    print('训练ECTSpeech模型（基于Easy Consistency Tuning）')
    # 检查teacher_model_path是否存在
    if not hasattr(params, 'teacher_model_path') or not params.teacher_model_path:
        raise ValueError("教师模型路径(teacher_model_path)未指定。请在params.py中设置teacher_model_path参数。")
    elif not os.path.exists(params.teacher_model_path):
        raise FileNotFoundError(f"教师模型文件不存在: {params.teacher_model_path}")
    else:
        print(f"将使用预训练教师模型: {params.teacher_model_path}")

    # 初始化模型
    model = Comospeech(nsymbols, 1, None, n_enc_channels, filter_channels, filter_channels_dp, 
                    n_heads, n_enc_layers, enc_kernel, enc_dropout, window_size, 
                    n_feats, teacher=False).cuda()
    
    # 使用与ECTSpeech相同的load_teacher_model函数加载教师模型
    model = load_teacher_model(model, checkpoint_dir=params.teacher_model_path)
    
    # 验证加载是否成功
    sample_params = []
    for name, param in model.decoder.denoise_fn.named_parameters():
        if len(sample_params) < 3:  # 只取样几个参数
            norm = torch.norm(param).item()
            mean = param.mean().item()
            sample_params.append((name, norm, mean))

    if sample_params:
        print("验证权重加载成功:")
        for name, norm, mean in sample_params:
            print(f"  - {name}: 范数={norm:.4f}, 均值={mean:.4f}")

        # 验证EMA参数是否匹配
        if hasattr(model.decoder, 'denoise_fn_ema'):
            ema_matched = True
            for name, param in model.decoder.denoise_fn.named_parameters():
                ema_param = dict(model.decoder.denoise_fn_ema.named_parameters())[name]
                if not torch.allclose(param, ema_param):
                    ema_matched = False
                    print(f"EMA参数不匹配: {name}")
            if ema_matched:
                print("✓ EMA参数正确复制")
    
    # 冻结编码器参数
    for param in model.encoder.parameters():
        param.requires_grad = False
    
    # 只优化解码器参数
    optimizer = torch.optim.Adam(model.decoder.denoise_fn.parameters(), lr=learning_rate)
    
    # 打印参数统计信息
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.denoise_fn.parameters())
    trainable_params = sum(p.numel() for p in model.decoder.denoise_fn.parameters() if p.requires_grad)
    print(f"模型参数统计: encoder = {encoder_params:,}参数 (已冻结), "
          f"decoder.denoise_fn = {decoder_params:,}参数 (可训练)")
    print(f"总可训练参数: {trainable_params:,}")

    # 在optimizer初始化后添加
    print(f"使用较小的初始学习率: {learning_rate}")
    print(f"最大梯度裁剪阈值: {max_grad_norm}")
    print(f"学习率下限: {min_lr}")
    print(f"损失缩放因子: {loss_scale}")
    print(f"模型初始阶段: {model.decoder.stage}")
    print(f"ECT比例: {model.decoder.ratio:.4f}")

    # 设置decoder的loss_scale参数
    model.decoder.loss_scale = loss_scale

    # 初始化迭代计数器和开始轮次 - 总是从头开始
    iter_count = 0
    start_epoch = 0
    
    # 删除检查点恢复代码
    # latest_checkpoint = ...
    
    print('记录测试批次...')
    test_batch = test_dataset.sample_test_batch(size=params.test_size)
    for i, item in enumerate(test_batch):
        mel = item['y']
        logger.add_image(f'image_{i}/ground_truth', plot_tensor(mel.squeeze()),
                         global_step=0, dataformats='HWC')
        save_plot(mel.squeeze(), f'{log_dir}/original_{i}.png')

    print(f'开始ECT微调，从epoch {start_epoch}、迭代 {iter_count} 开始...')
    model.train()
    
    # 记录连续NaN批次计数
    nan_batches_count = 0
    max_nan_batches = 5  # 如果连续5个批次出现NaN，则警告并考虑降低学习率
    
    # 添加学习率状态跟踪
    lr_reduced_times = 0  # 记录学习率被降低的次数
    max_lr_reduce = 3     # 最多降低几次学习率
    
    # 根据每个epoch的批次数和double_every参数计算正确的iter_d值
    correct_iter_d = steps_per_epoch * double_every

    # 读取最小阶段长度参数
    min_stage_length = getattr(params, 'ect_min_stage_length', 3000)

    # 确保iter_d不小于最小阶段长度
    if correct_iter_d < min_stage_length:
        print(f"警告: 计算出的iter_d={correct_iter_d}小于最小阶段长度{min_stage_length}")
        correct_iter_d = min_stage_length
        print(f"已将iter_d调整为最小值{min_stage_length}（约{min_stage_length/steps_per_epoch:.1f}个epoch）")

    # 设置模型的iter_d参数，确保每double_every个epoch才完成一个阶段
    model.decoder.iter_d = correct_iter_d
    model.decoder.steps_per_epoch = steps_per_epoch
    print(f"根据数据集大小调整iter_d={correct_iter_d}（约{correct_iter_d/steps_per_epoch:.1f}个epoch完成一个阶段）")
    
    # 如果从检查点恢复训练，重新计算当前ratio
    if iter_count > 0:
        # 使用新的iter_d重新计算阶段和比例
        model.decoder.iter_count = iter_count
        stage = iter_count // correct_iter_d
        progress_in_stage = (iter_count % correct_iter_d) / correct_iter_d
        
        # 计算base_ratio和next_ratio
        if stage == 0:
            base_ratio = 0.0
        else:
            base_ratio = 1 - 1 / (model.decoder.q ** stage)
            
        if stage == 0:
            next_ratio = 1 - 1 / model.decoder.q
        else:
            next_ratio = 1 - 1 / (model.decoder.q ** (stage+1))
        
        # 平滑过渡
        if hasattr(model.decoder, 'loss_fn'):
            model.decoder.loss_fn.ratio = base_ratio + (next_ratio - base_ratio) * progress_in_stage
            model.decoder.loss_fn.stage = stage
            
            print(f"根据iter_count={iter_count}重新计算: stage={stage}, ratio={model.decoder.ratio:.4f}")
        else:
            print(f"警告: 模型没有loss_fn属性，无法设置stage={stage}和ratio")

    for epoch in range(start_epoch, n_epochs + 1):
        # 每个epoch开始时根据之前的NaN情况进行调试级别调整
        if nan_batches_count > 0:
            set_debug_level(True)  # 有NaN时开启详细日志
        elif epoch % 10 == 0:  # 每10个epoch开启一次详细日志
            set_debug_level(True)
        else:
            set_debug_level(False)  # 正常情况下使用简洁日志
            
        # 打印当前迭代次数和比例
        current_ratio = model.decoder.ratio
        print(f'Epoch {epoch}: 当前迭代 {iter_count}，ECT比例为 {current_ratio:.4f}')
            
        dur_losses = []
        prior_losses = []
        diff_losses = []
        
        # 为了跟踪每个epoch中NaN出现的频率
        nan_count = 0
        total_batches = 0
        
        with tqdm(loader, total=len(loader)) as progress_bar:
            for batch_idx, batch in enumerate(progress_bar):
                total_batches += 1
                
                try:
                    model.zero_grad()
                    x, x_lengths = batch['x'].cuda(), batch['x_lengths'].cuda()
                    y, y_lengths = batch['y'].cuda(), batch['y_lengths'].cuda()
                    
                    # 检查输入数据是否有NaN
                    if not (check_tensor(x, "输入x") and check_tensor(y, "输入y")):
                        nan_count += 1
                        nan_batches_count += 1
                        continue
                    
                    dur_loss, prior_loss, diff_loss = model.compute_loss(x, x_lengths,
                                                                        y, y_lengths,
                                                                        out_size=out_size)
                    
                    # 检查损失值是否有效
                    valid_dur = check_tensor(dur_loss, "dur_loss")
                    valid_prior = check_tensor(prior_loss, "prior_loss")
                    valid_diff = check_tensor(diff_loss, "diff_loss")
                    
                    if not (valid_dur and valid_prior and valid_diff):
                        nan_count += 1
                        nan_batches_count += 1
                        
                        # 如果连续多个批次出现NaN，考虑降低学习率
                        if nan_batches_count >= max_nan_batches:
                            print(f"警告: 连续 {nan_batches_count} 个批次出现NaN。降低学习率...")
                            
                            if lr_reduced_times < max_lr_reduce:
                                # 根据策略降低学习率
                                new_lr = adjust_learning_rate(optimizer, lr_factor=0.5, min_lr=min_lr)
                                lr_reduced_times += 1
                                print(f"学习率已降低至 {new_lr:.6f}，({lr_reduced_times}/{max_lr_reduce})")
                            
                            nan_batches_count = 0  # 重置计数器
                            
                        continue
                    
                    nan_batches_count = 0  # 重置计数器，因为这个批次是有效的
                    
                    # ECT模式下只使用diff_loss
                    loss = diff_loss
                        
                    # 检查最终损失
                    if not check_tensor(loss, "最终loss"):
                        nan_count += 1
                        continue
                    
                    # 反向传播
                    loss.backward()
                    
                    # 检查梯度是否包含NaN
                    has_nan_grad = False
                    for name, param in model.decoder.denoise_fn.named_parameters():
                        if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                            print(f"警告: {name} 的梯度包含NaN/Inf值")
                            has_nan_grad = True
                            break

                    if has_nan_grad:
                        print("检测到NaN梯度，跳过此批次更新")
                        nan_count += 1
                        nan_batches_count += 1
                        continue
                    
                    # 梯度裁剪
                    dec_grad_norm = torch.nn.utils.clip_grad_norm_(model.decoder.denoise_fn.parameters(),
                                                                max_norm=max_grad_norm)
                    
                    # 应用优化器步骤
                    optimizer.step()
                    
                    # 更新EMA模型和迭代计数
                    model.decoder.update_ema()

                    # 更新迭代计数器（只增加局部变量)
                    iter_count += 1

                    # 检查是否需要根据迭代数更新阶段和比例
                    if iter_count % correct_iter_d == 0:
                        new_stage = iter_count // correct_iter_d
                        print(f"达到阶段转换点! 更新阶段: {new_stage}")
                        model.decoder.update_schedule(new_stage)
                    
                    # 记录有效的损失值
                    logger.add_scalar('training/duration_loss', dur_loss.item(),
                                        global_step=iter_count)
                    logger.add_scalar('training/prior_loss', prior_loss.item(),
                                        global_step=iter_count)
                    logger.add_scalar('training/diffusion_loss', diff_loss.item(),
                                        global_step=iter_count)
                    
                    # 记录解码器梯度
                    logger.add_scalar('training/decoder_grad_norm', dec_grad_norm, global_step=iter_count)
                    
                    # 记录学习率
                    logger.add_scalar('training/learning_rate', optimizer.param_groups[0]['lr'],
                                        global_step=iter_count)
                    
                    # 记录ECT的比例信息
                    logger.add_scalar('training/ect_ratio', model.decoder.ratio,
                                        global_step=iter_count)
                    
                    dur_losses.append(dur_loss.item())
                    prior_losses.append(prior_loss.item())
                    diff_losses.append(diff_loss.item())
                    
                    # 更新进度条
                    if batch_idx % 5 == 0:
                        msg = f'Epoch: {epoch}, iteration: {iter_count} | diff_loss: {diff_loss.item():.6f}'
                        msg += f' | ect_ratio: {model.decoder.ratio:.4f}'
                        if nan_count > 0:
                            msg += f' | NaN批次: {nan_count}/{total_batches}'
                        progress_bar.set_description(msg)
                    
                except Exception as e:
                    print(f"处理批次 {batch_idx} 时出错: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    nan_count += 1
                    continue

        # 检查是否有足够的有效损失来计算平均值
        if len(dur_losses) > 0 and len(prior_losses) > 0 and len(diff_losses) > 0:
            log_msg = 'Epoch %d: diffusion loss = %.3f' % (epoch, np.mean(diff_losses))
            log_msg += ' | ect_ratio = %.4f' % (model.decoder.ratio)
            
            # 添加NaN批次和学习率信息
            log_msg += f' | NaN批次: {nan_count}/{total_batches} | lr = {optimizer.param_groups[0]["lr"]:.6f}'
            
            # 🆕 新增：计算验证集损失（仅扩散损失）
            print('计算验证集损失...')
            val_diff, val_count = calculate_validation_loss(model, validation_loader, out_size)
            
            if val_diff is not None:
                log_msg += f' | val_loss = {val_diff:.3f}'
                
                # 记录到TensorBoard
                logger.add_scalar('validation/diffusion_loss', val_diff, global_step=iter_count)
                
                print(f'验证集扩散损失: {val_diff:.3f} (基于{val_count}个有效批次)')
            else:
                log_msg += ' | val_loss = N/A'
                print('验证集损失计算失败')
            
            # 写入日志文件
            with open(f'{log_dir}/train.log', 'a') as f:
                f.write(log_msg + '\n')
                
            # 打印到控制台
            print(log_msg)
        else:
            print(f"警告: epoch {epoch} 没有有效的损失值，无法计算平均值")

        # 如果NaN比例过高，考虑降低学习率
        if total_batches > 0 and nan_count / total_batches > 0.3:  # 降低阈值到30%
            print(f"警告: epoch {epoch} 中 {nan_count}/{total_batches} 批次出现NaN (超过30%)，降低学习率")
            
            if lr_reduced_times < max_lr_reduce:
                adjust_learning_rate(optimizer, lr_factor=0.5, min_lr=min_lr)
                lr_reduced_times += 1
                print(f"学习率已被降低 {lr_reduced_times}/{max_lr_reduce} 次")
            else:
                print(f"已达到最大学习率调整次数 ({max_lr_reduce})，不再降低学习率")

        # 每个epoch结束时保存最新模型
        try:
            save_path = f"{log_dir}/ect_model_{epoch}.pt"
            save_model(model, optimizer, iter_count, save_path, epoch)
        except Exception as e:
            print(f"保存模型时出错: {str(e)}")
            # 尝试使用不同的格式保存
            try:
                save_path = f"{log_dir}/ect_model_{epoch}_backup.pt"
                torch.save({
                    'model': model.state_dict(),
                    'iter_count': iter_count,
                    'epoch': epoch
                }, save_path, _use_new_zipfile_serialization=False)
                print(f'模型已使用备用方式保存: {save_path}')
            except:
                print("无法保存模型，跳过此步骤")
                
        model.train()

        if epoch % params.save_every > 0:
            continue

        model.eval()
        print('生成合成音频...')
        with torch.no_grad():
            for i, item in enumerate(test_batch):
                try:
                    x = item['x'].to(torch.long).unsqueeze(0).cuda()
                    x_lengths = torch.LongTensor([x.shape[-1]]).cuda()
                    
                    # ECT模式用1步采样并使用EMA模型
                    y_enc, y_dec, attn = model(x, x_lengths, n_timesteps=1, use_ema=True)
                    
                    # 检查生成的结果是否有效
                    if not (check_tensor(y_enc, "y_enc") and check_tensor(y_dec, "y_dec") and check_tensor(attn, "attn")):
                        print(f"样本 {i} 生成结果包含NaN/Inf，跳过记录")
                        continue
                    
                    # 记录生成结果
                    logger.add_image(f'image_{i}/generated_enc',
                                   plot_tensor(y_enc.squeeze().cpu()),
                                        global_step=iter_count, dataformats='HWC')
                    logger.add_image(f'image_{i}/generated_dec',
                                   plot_tensor(y_dec.squeeze().cpu()),
                                        global_step=iter_count, dataformats='HWC')
                    logger.add_image(f'image_{i}/alignment',
                                   plot_tensor(attn.squeeze().cpu()),
                                        global_step=iter_count, dataformats='HWC')
                    save_plot(y_enc.squeeze().cpu(), 
                           f'{log_dir}/generated_enc_{i}.png')
                    save_plot(y_dec.squeeze().cpu(), 
                           f'{log_dir}/generated_dec_{i}.png')
                    save_plot(attn.squeeze().cpu(), 
                           f'{log_dir}/alignment_{i}.png')
                except Exception as e:
                    print(f"生成样本 {i} 时出错: {str(e)}")
                    continue

    # 在模型加载后的验证部分添加更详细的参数检查
    print("\n初始化模型参数检查:")
    print(f"  - P_mean: {model.decoder.P_mean}")
    print(f"  - P_std: {model.decoder.P_std}")
    print(f"  - q: {model.decoder.q}")
    print(f"  - adap_c: {model.decoder.adap_c}")
    print(f"  - loss_scale: {model.decoder.loss_scale}")
    print(f"  - iter_d: {model.decoder.iter_d}")
    print(f"  - steps_per_epoch: {steps_per_epoch}")
    print(f"  - sigma_min: {model.decoder.sigma_min}")
    print(f"  - sigma_max: {model.decoder.sigma_max}")