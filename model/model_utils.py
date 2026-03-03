import torch
import copy
import os
import sys

def check_ect_params():
    """检查是否存在ECT相关参数
    
    Returns:
        tuple: (has_params, params对象)
    """
    try:
        import params
        has_ect_params = hasattr(params, 'ect_eps')
        return has_ect_params, params
    except ImportError:
        print("警告: 未找到params模块，将使用默认值")
        return False, None

def load_denoise_weights(model, state_dict, verbose=False):
    """从教师模型加载denoise_fn权重
    
    Args:
        model: 目标模型
        state_dict: 教师模型的状态字典
        verbose: 是否输出详细信息
        
    Returns:
        bool: 加载是否成功
    """
    # 提取状态字典
    if 'model' in state_dict:
        state_dict = state_dict['model']
    elif 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    
    # 提取教师模型的denoise_fn参数
    denoise_fn_params = {}
    denoise_fn_prefix = 'decoder.denoise_fn.'
    
    # 查找所有denoise_fn相关参数
    for k in state_dict.keys():
        if denoise_fn_prefix in k:
            rel_key = k.split(denoise_fn_prefix)[1]
            denoise_fn_params[rel_key] = state_dict[k]
    
    # 如果找不到denoise_fn参数，则尝试直接加载
    if len(denoise_fn_params) == 0:
        if verbose:
            print("在教师模型中未找到denoise_fn参数，尝试直接加载...")
        
        # 获取当前模型参数的键
        current_keys = set()
        for name, _ in model.denoise_fn.named_parameters():
            current_keys.add(name)
        
        # 尝试匹配教师模型中的参数
        for k in state_dict.keys():
            key_parts = k.split('.')
            if len(key_parts) >= 2 and key_parts[-2] + '.' + key_parts[-1] in current_keys:
                rel_key = key_parts[-2] + '.' + key_parts[-1]
                denoise_fn_params[rel_key] = state_dict[k]
    
    # 验证参数数量与名称
    current_param_count = sum(1 for _ in model.denoise_fn.parameters())
    
    if len(denoise_fn_params) == 0:
        print("错误: 未找到任何可匹配的denoise_fn参数")
        return False
    
    if len(denoise_fn_params) != current_param_count:
        print(f"错误: 参数数量不匹配，当前模型={current_param_count}，教师模型={len(denoise_fn_params)}")
        return False
    
    # 加载参数
    missing_keys, unexpected_keys = model.denoise_fn.load_state_dict(denoise_fn_params, strict=False)
    
    if missing_keys or unexpected_keys:
        if missing_keys:
            print(f"错误: 缺失的键: {missing_keys}")
        if unexpected_keys:
            print(f"错误: 未预期的键: {unexpected_keys}")
        return False
    
    # 重新初始化EMA模型，使其与主模型一致
    model.denoise_fn_ema = copy.deepcopy(model.denoise_fn)
    # 禁用EMA模型的梯度
    for param in model.denoise_fn_ema.parameters():
        param.requires_grad = False
    
    # 设置初始阶段
    model.stage = 0
    model.ratio = 0.0
    
    print(f"成功加载教师模型权重，设置初始阶段={model.stage}，比例={model.ratio:.4f}")
    print("重新初始化EMA模型...")
    print("ComoECT模型denoise_fn权重加载成功!")
    
    return True

# 调试日志控制全局变量
DEBUG_VERBOSE = False  # 默认不显示详细日志

def set_debug_level(level):
    """设置全局日志详细程度
    
    Args:
        level: 布尔值，True启用详细日志，False关闭
    """
    global DEBUG_VERBOSE  # 声明使用全局变量
    DEBUG_VERBOSE = level  # 更新全局日志级别
    # 同步到como_ect模块  # 确保日志级别在其他模块中同步
    try:
        import model.como_ect  # 尝试导入como_ect模块
        model.como_ect.DEBUG_VERBOSE = level  # 更新como_ect模块中的日志级别
    except:
        pass  # 如果导入失败，静默处理异常

def debug_print(*args, **kwargs):
    """条件打印函数，仅在详细日志模式下输出信息
    
    Args:
        *args: 要打印的参数
        **kwargs: 打印函数的关键字参数
    """
    if DEBUG_VERBOSE:  # 检查是否启用了详细日志
        print(*args, **kwargs)  # 在详细日志模式下打印信息

def check_tensor(tensor, name=""):
    """检查张量是否有效（无NaN和Inf）
    
    Args:
        tensor: 要检查的张量
        name: 张量名称，用于日志输出
    
    Returns:
        bool: 如果张量有效返回True，否则返回False
    """
    if tensor is None:  # 检查张量是否为None
        return False  # 如果是None则返回无效
    is_valid = not torch.isnan(tensor).any() and not torch.isinf(tensor).any()  # 检查是否包含NaN或Inf
    if not is_valid and DEBUG_VERBOSE:  # 如果无效且启用了详细日志
        print(f"警告: {name} 包含NaN或Inf值")  # 输出警告信息
    return is_valid  # 返回检查结果

def handle_nan_batches(nan_batches_count, max_nan_batches, lr_reduced_times, max_lr_reduce, optimizer, min_lr):
    """处理连续NaN批次，根据需要调整学习率
    
    Args:
        nan_batches_count: 连续NaN批次计数
        max_nan_batches: 触发学习率调整的最大连续NaN批次数
        lr_reduced_times: 学习率已降低的次数
        max_lr_reduce: 最大学习率降低次数
        optimizer: 优化器对象
        min_lr: 最小学习率
    
    Returns:
        tuple: (new_nan_batches_count, new_lr_reduced_times, lr_adjusted)
    """
    # 检查是否需要降低学习率
    if nan_batches_count >= max_nan_batches:
        print(f"警告: 连续 {nan_batches_count} 个批次出现NaN。考虑降低学习率...")
        
        if lr_reduced_times < max_lr_reduce:
            # 根据策略降低学习率
            new_lr = adjust_learning_rate(optimizer, lr_factor=0.5, min_lr=min_lr)
            lr_reduced_times += 1
            print(f"学习率已降低至 {new_lr:.6f}，({lr_reduced_times}/{max_lr_reduce})")
            return 0, lr_reduced_times, True  # 重置NaN计数器，返回调整后的状态
        else:
            print(f"已达到最大学习率调整次数 ({max_lr_reduce})，不再降低学习率")
    
    return nan_batches_count, lr_reduced_times, False  # 返回原始状态

def adjust_learning_rate(optimizer, lr_factor=0.5, min_lr=1e-6):
    """根据指定因子降低学习率，但确保不低于最小学习率
    
    Args:
        optimizer: 优化器对象
        lr_factor: 学习率降低因子
        min_lr: 最小学习率
    
    Returns:
        float: 调整后的学习率
    """
    current_lr = optimizer.param_groups[0]['lr']
    new_lr = max(current_lr * lr_factor, min_lr)
    
    if new_lr < current_lr:
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        print(f"学习率调整: {current_lr:.6f} -> {new_lr:.6f}")
    else:
        print(f"学习率已达最小值: {current_lr:.6f}，不再降低")
    
    return new_lr

def handle_epoch_nan_stats(nan_count, total_batches, threshold, lr_reduced_times, max_lr_reduce, optimizer, min_lr):
    """处理每个epoch结束时的NaN统计，并在需要时调整学习率
    
    Args:
        nan_count: NaN批次数量
        total_batches: 总批次数量
        threshold: 触发学习率调整的NaN比例阈值
        lr_reduced_times: 学习率已降低的次数
        max_lr_reduce: 最大学习率降低次数
        optimizer: 优化器对象
        min_lr: 最小学习率
        
    Returns:
        int: 更新后的lr_reduced_times
    """
    # 检查NaN比例是否超过阈值
    if total_batches > 0 and nan_count / total_batches > threshold:
        print(f"警告: 本epoch中 {nan_count}/{total_batches} 批次出现NaN (超过{threshold*100}%)，降低学习率")
        
        if lr_reduced_times < max_lr_reduce:
            adjust_learning_rate(optimizer, lr_factor=0.5, min_lr=min_lr)
            lr_reduced_times += 1
            print(f"学习率已被降低 {lr_reduced_times}/{max_lr_reduce} 次")
        else:
            print(f"已达到最大学习率调整次数 ({max_lr_reduce})，不再降低学习率")
            
    return lr_reduced_times

def save_model(model, epoch, log_dir, teacher=False, verbose=True):
    """保存模型到指定路径
    
    Args:
        model: 要保存的模型
        epoch: 当前训练轮次
        log_dir: 日志目录路径
        teacher: 是否为教师模型
        verbose: 是否输出详细信息
    
    Returns:
        bool: 保存是否成功
    """
    try:
        ckpt = model.state_dict()
        model_prefix = "teacher" if teacher else "ect"
        save_path = f"{log_dir}/{model_prefix}_model_{epoch}.pt"
        torch.save(ckpt, f=save_path)
        if verbose:
            print(f'模型已保存: {save_path}')
        return True
    except Exception as e:
        print(f"保存模型时出错: {str(e)}")
        return False
