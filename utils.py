import os
import copy
import re
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt
import numpy as np
import torch


def intersperse(lst, item):
    # Adds blank symbol
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


def parse_filelist(filelist_path, split_char="|"):
    with open(filelist_path, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split_char) for line in f]
    return filepaths_and_text


def latest_checkpoint_path(dir_path, regex="grad_*.pt"):
    f_list = glob.glob(os.path.join(dir_path, regex))
    f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    x = f_list[-1]
    return x


def load_checkpoint(logdir, model, num=None):
    if num is None:
        model_path = latest_checkpoint_path(logdir, regex="grad_*.pt")
    else:
        model_path = os.path.join(logdir, f"grad_{num}.pt")
    print(f'Loading checkpoint {model_path}...')
    model_dict = torch.load(model_path, map_location=lambda loc, storage: loc)
    model.load_state_dict(model_dict, strict=False)
    return model


def save_figure_to_numpy(fig):
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


def plot_tensor(tensor):
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(tensor, aspect="auto", origin="lower", interpolation='none')
    plt.axis('off')
    # plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data


def save_plot(tensor, savepath):
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(tensor, aspect="auto", origin="lower", interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.canvas.draw()
    plt.savefig(savepath)
    plt.close()
    return

def load_teacher_model(model, checkpoint_dir):
    """
    加载预训练的教师模型参数到当前模型
    
    对于ECT方法，只需加载模型参数，然后初始化微调阶段
    """
    print(f'加载预训练教师模型: {checkpoint_dir}')
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"错误: 教师模型文件不存在于路径 {checkpoint_dir}")
    
    try:
        # 加载模型状态字典
        checkpoint = torch.load(checkpoint_dir, map_location=lambda loc, storage: loc)
        
        # 获取状态字典（兼容不同的保存格式）
        state_dict = None
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
                print("从checkpoint的'model'键中提取权重")
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                print("从checkpoint的'state_dict'键中提取权重")
            else:
                # 假设整个checkpoint就是状态字典
                state_dict = checkpoint
                print("将整个checkpoint作为状态字典处理")
        else:
            # 不是字典，可能直接是模型参数
            state_dict = checkpoint
            print("直接使用checkpoint作为状态字典")
        
        if state_dict:
            print(f"状态字典包含{len(state_dict)}个键")
            
            # 特殊处理ComoECT模型
            if hasattr(model, 'decoder') and hasattr(model.decoder, 'load_teacher_weights'):
                print("\n检测到ComoECT模型，使用专用方法加载denoise_fn权重...")
                # 提取denoise_fn相关参数
                denoise_fn_dict = {}
                prefix = 'decoder.denoise_fn.'
                
                for k, v in state_dict.items():
                    if k.startswith(prefix):
                        # 去掉前缀，只保留denoise_fn内部结构
                        new_key = k[len(prefix):]
                        denoise_fn_dict[new_key] = v
                
                if len(denoise_fn_dict) > 0:
                    print(f"成功提取了{len(denoise_fn_dict)}个denoise_fn参数")
                    # 使用专用方法加载
                    success = model.decoder.load_teacher_weights(denoise_fn_dict, verbose=True)
                    
                    # 在加载后手动初始化stage和ratio
                    if hasattr(model.decoder, 'loss_fn'):
                        model.decoder.loss_fn.stage = 0
                        model.decoder.loss_fn.ratio = 0.0
                        print(f"初始化ECT状态: stage={model.decoder.stage}, ratio={model.decoder.ratio:.4f}")
                    else:
                        print("警告: 模型没有loss_fn属性，无法初始化stage和ratio")
                    
                    # 初始化EMA权重
                    print("重置EMA权重...")
                    model.decoder.denoise_fn_ema.load_state_dict(model.decoder.denoise_fn.state_dict())
                    print("✓ EMA模型与主模型同步")
                    
                if success:
                    print("ComoECT模型denoise_fn权重加载成功!")
                else:
                    print("警告: ComoECT模型denoise_fn权重加载问题")
            else:
                print("找不到denoise_fn相关参数，尝试其他加载方式")
            
            # 尝试直接加载完整模型权重
            try:
                print("\n尝试加载完整模型权重...")
                incompatible_keys = model.load_state_dict(state_dict, strict=False)
                
                # 打印未加载的键
                if incompatible_keys.missing_keys:
                    print(f"未加载的键: {len(incompatible_keys.missing_keys)}个")
                if incompatible_keys.unexpected_keys:
                    print(f"多余的键: {len(incompatible_keys.unexpected_keys)}个")
                
                print("完整模型权重加载完成（使用非严格模式）")
            except Exception as e:
                print(f"加载完整模型权重时出错: {str(e)}")
        else:
            print("警告: 无法获取有效的状态字典")
            analyze_teacher_checkpoint(checkpoint_dir)
        
        # 验证加载后的状态
        # 检查参数是否变化
        param_stats = []
        for name, param in model.named_parameters():
            if 'denoise_fn' in name:
                param_stats.append((name, param.data.mean().item(), param.data.std().item()))
                if len(param_stats) >= 5:
                    break
                    
        print("\n加载后的参数统计:")
        for name, mean, std in param_stats:
            print(f"  {name}: mean={mean:.6f}, std={std:.6f}")
            
        # 验证编码器参数
        encoder_params = [p for name, p in model.named_parameters() if 'encoder' in name]
        if encoder_params:
            encoder_mean = sum(p.mean().item() for p in encoder_params) / len(encoder_params)
            print(f"编码器参数平均值: {encoder_mean:.6f}")
        
        # 如果是ComoECT模型，确保EMA模型与主模型一致
        if hasattr(model, 'decoder') and hasattr(model.decoder, 'denoise_fn_ema'):
            # 强制重新初始化EMA模型，确保与主模型完全一致
            model.decoder.denoise_fn_ema = copy.deepcopy(model.decoder.denoise_fn)
            # 禁用EMA模型的梯度
            for param in model.decoder.denoise_fn_ema.parameters():
                param.requires_grad = False
            print("✓ 已重新初始化EMA模型，确保与主模型完全一致")
        
        return model
            
    except Exception as e:
        print(f"加载教师模型时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\n教师模型加载失败! 退出程序以避免训练错误。")
        import sys
        sys.exit(1)  # 退出程序

def analyze_teacher_checkpoint(checkpoint_path):
    """
    详细分析教师模型的结构和参数，用于诊断问题
    """
    print(f"正在分析教师模型: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=lambda loc, storage: loc)
        
        # 分析checkpoint类型
        print(f"Checkpoint类型: {type(checkpoint)}")
        
        if isinstance(checkpoint, dict):
            # 统计一级键
            top_keys = list(checkpoint.keys())
            print(f"顶级键: {top_keys}")
            
            # 寻找可能的模型状态字典
            state_dict = None
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
                print("从'model'键中提取状态字典")
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict'] 
                print("从'state_dict'键中提取状态字典")
            elif any(k.endswith('.weight') for k in top_keys):
                state_dict = checkpoint
                print("checkpoint本身似乎是一个状态字典")
            
            # 如果找到状态字典，分析它的结构
            if state_dict:
                all_keys = list(state_dict.keys())
                print(f"状态字典包含{len(all_keys)}个键")
                
                # 分析前缀分布
                prefixes = {}
                for k in all_keys:
                    parts = k.split('.')
                    prefix = parts[0] if parts else k
                    prefixes[prefix] = prefixes.get(prefix, 0) + 1
                
                print("键前缀分布:")
                for prefix, count in sorted(prefixes.items(), key=lambda x: x[1], reverse=True):
                    print(f"  - {prefix}: {count}个键")
                
                # 分析一些键的形状
                print("\n参数形状示例:")
                for i, k in enumerate(all_keys):
                    if i < 10:  # 只显示前10个
                        shape = state_dict[k].shape if hasattr(state_dict[k], 'shape') else None
                        print(f"  - {k}: {shape}")
                    else:
                        break
                
                # 特别查找denoise_fn相关的键
                denoise_keys = [k for k in all_keys if 'denoise_fn' in k]
                print(f"\n找到{len(denoise_keys)}个包含'denoise_fn'的键")
                if denoise_keys:
                    # 分析这些键的模式
                    denoise_prefixes = set()
                    for k in denoise_keys:
                        parts = k.split('.')
                        if len(parts) > 1 and 'denoise_fn' in parts:
                            idx = parts.index('denoise_fn')
                            prefix = '.'.join(parts[:idx+1])
                            denoise_prefixes.add(prefix)
                    
                    print(f"denoise_fn键的前缀模式: {denoise_prefixes}")
                    
                    # 显示一些denoise_fn键的示例
                    print("\ndenoise_fn键示例:")
                    for i, k in enumerate(denoise_keys):
                        if i < 10:
                            print(f"  - {k}")
                        else:
                            print(f"  - ... 以及其他{len(denoise_keys)-10}个键")
                            break
                else:
                    # 查找其他可能的模型结构键
                    print("\n未找到denoise_fn键，查找其他可能的模型结构...")
                    model_keys = [k for k in all_keys if any(x in k for x in ['encoder', 'decoder', 'conv', 'layer', 'block', 'net'])]
                    if model_keys:
                        print(f"找到{len(model_keys)}个可能的模型结构键")
                        print("示例:")
                        for i, k in enumerate(model_keys):
                            if i < 10:
                                print(f"  - {k}")
                            else:
                                break
            else:
                print("警告: 无法识别状态字典结构")
                
                # 尝试分析checkpoint的其他部分
                print("\n分析checkpoint的其他键:")
                for k, v in checkpoint.items():
                    if isinstance(v, dict):
                        print(f"  - {k}: 字典，包含{len(v)}个键")
                    elif isinstance(v, torch.Tensor):
                        print(f"  - {k}: Tensor，形状为{v.shape}")
                    else:
                        print(f"  - {k}: {type(v)}")
        else:
            print("警告: checkpoint不是字典类型，无法进一步分析")
    
    except Exception as e:
        print(f"分析教师模型时出错: {str(e)}")
        import traceback
        traceback.print_exc()

def check_model_params(checkpoint_path, model=None, check_params=True):
    """
    详细检查预训练模型与当前模型的参数
    
    Args:
        checkpoint_path: 预训练模型的路径
        model: 当前模型实例（可选）
        check_params: 是否检查参数形状匹配（如果传入了model）
    """
    print(f"\n详细检查模型参数: {checkpoint_path}")
    
    try:
        # 加载checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 确定checkpoint的类型
        if isinstance(checkpoint, dict):
            state_dict = None
            
            # 提取状态字典
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
                print(f"使用 'model' 键中的状态字典，包含 {len(state_dict)} 个参数")
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                print(f"使用 'state_dict' 键中的状态字典，包含 {len(state_dict)} 个参数")
            else:
                # 假设整个checkpoint就是状态字典
                state_dict = checkpoint
                print(f"使用整个checkpoint作为状态字典，包含 {len(state_dict)} 个参数")
                
            # 分析键前缀
            prefixes = {}
            for key in state_dict.keys():
                parts = key.split('.')
                prefix = parts[0] if len(parts) > 0 else key
                if prefix not in prefixes:
                    prefixes[prefix] = 0
                prefixes[prefix] += 1
                
            print("\n状态字典前缀分布:")
            for prefix, count in prefixes.items():
                print(f"  {prefix}: {count}个键")
                
            # 检查denoise_fn
            denoise_keys = []
            
            # 检查各种可能的denoise_fn路径
            patterns = [
                'denoise_fn.',
                'decoder.denoise_fn.'
            ]
            
            for pattern in patterns:
                pattern_keys = [k for k in state_dict.keys() if pattern in k]
                if pattern_keys:
                    print(f"\n找到 {len(pattern_keys)} 个包含 '{pattern}' 的键")
                    if len(pattern_keys) > 0:
                        print(f"示例: {pattern_keys[:3]}")
                    denoise_keys.extend(pattern_keys)
            
            # 检查参数与当前模型匹配情况
            if model is not None and check_params:
                print("\n检查参数与当前模型的匹配情况:")
                
                # 获取当前模型的denoise_fn参数
                if hasattr(model, 'decoder') and hasattr(model.decoder, 'denoise_fn'):
                    target_params = {name: param for name, param in model.decoder.denoise_fn.named_parameters()}
                    
                    # 检查预训练权重是否与当前模型匹配
                    for pattern in patterns:
                        matched_params = 0
                        for target_name, target_param in target_params.items():
                            source_key = f"{pattern}{target_name}"
                            if source_key in state_dict:
                                source_param = state_dict[source_key]
                                if source_param.shape == target_param.shape:
                                    matched_params += 1
                                else:
                                    print(f"形状不匹配: {source_key} {source_param.shape} vs {target_name} {target_param.shape}")
                        
                        if matched_params > 0:
                            print(f"使用前缀 '{pattern}' 能匹配 {matched_params}/{len(target_params)} 个参数")
                    
                    # 尝试一些不同的路径组合
                    for prefix_pattern in ['decoder.', '', 'model.decoder.']:
                        matched_params = 0
                        for target_name, target_param in target_params.items():
                            source_key = f"{prefix_pattern}denoise_fn.{target_name}"
                            if source_key in state_dict:
                                source_param = state_dict[source_key]
                                if source_param.shape == target_param.shape:
                                    matched_params += 1
                        
                        if matched_params > 0:
                            print(f"使用前缀 '{prefix_pattern}denoise_fn.' 能匹配 {matched_params}/{len(target_params)} 个参数")
                
            # 输出权重的一些统计信息
            print("\n权重统计信息:")
            param_stats = {}
            for k, v in state_dict.items():
                if isinstance(v, torch.Tensor) and v.numel() > 0:
                    try:
                        mean_val = v.mean().item()
                        std_val = v.std().item()
                        param_stats[k] = (mean_val, std_val)
                    except:
                        pass
            
            # 打印一些权重的均值和标准差
            if param_stats:
                samples = list(param_stats.items())[:5]
                for k, (mean, std) in samples:
                    print(f"  {k}: mean={mean:.6f}, std={std:.6f}")
        
        else:
            print("Checkpoint不是字典类型，无法分析")
    
    except Exception as e:
        print(f"检查模型参数时出错: {str(e)}")
        import traceback
        traceback.print_exc()

def print_model_structure(model, max_depth=None):
    """
    打印模型的结构以分析参数层次
    
    Args:
        model: PyTorch模型
        max_depth: 最大打印深度，None表示打印所有层次
    """
    def _print_module(module, prefix='', depth=0):
        if max_depth is not None and depth > max_depth:
            return
            
        for name, child in module.named_children():
            child_prefix = f"{prefix}.{name}" if prefix else name
            print(f"{'  ' * depth}[{child_prefix}] -> {child.__class__.__name__}")
            _print_module(child, child_prefix, depth + 1)
            
        # 如果是叶子模块，打印参数
        if sum(1 for _ in module.named_children()) == 0:
            for param_name, param in module.named_parameters(recurse=False):
                full_param_name = f"{prefix}.{param_name}" if prefix else param_name
                print(f"{'  ' * depth}  └─ {full_param_name}: {param.shape}")
    
    print(f"\n模型结构分析 - {model.__class__.__name__}:")
    _print_module(model)
    
    # 打印总参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print(f"固定参数: {total_params - trainable_params:,}")

def check_model_compatibility(model, state_dict):
    """
    详细检查模型结构与预训练权重的匹配情况
    特别关注encoder部分的匹配度
    
    Args:
        model: 当前模型实例
        state_dict: 预训练模型的状态字典
    
    Returns:
        dict: 包含匹配统计的字典
    """
    # 统计和分类所有键
    all_keys = list(state_dict.keys())
    encoder_keys = [k for k in all_keys if k.startswith('encoder')]
    decoder_keys = [k for k in all_keys if k.startswith('decoder')]
    other_keys = [k for k in all_keys if not k.startswith(('encoder', 'decoder'))]
    
    # 获取当前模型的所有参数键
    model_keys = []
    encoder_model_keys = []
    decoder_model_keys = []
    
    for name, _ in model.named_parameters():
        model_keys.append(name)
        if name.startswith('encoder'):
            encoder_model_keys.append(name)
        elif name.startswith('decoder'):
            decoder_model_keys.append(name)
    
    # 计算匹配统计
    # 1. 预训练模型中有但当前模型没有的键（多余键）
    extra_keys = [k for k in all_keys if k not in model_keys]
    extra_encoder_keys = [k for k in encoder_keys if k not in encoder_model_keys]
    extra_decoder_keys = [k for k in decoder_keys if k not in decoder_model_keys]
    
    # 2. 当前模型有但预训练模型没有的键（缺失键）
    missing_keys = [k for k in model_keys if k not in all_keys]
    missing_encoder_keys = [k for k in encoder_model_keys if k not in encoder_keys]
    missing_decoder_keys = [k for k in decoder_model_keys if k not in decoder_keys]
    
    # 3. 两者都有的键（匹配键）
    matched_keys = [k for k in model_keys if k in all_keys]
    matched_encoder_keys = [k for k in encoder_model_keys if k in encoder_keys]
    matched_decoder_keys = [k for k in decoder_model_keys if k in decoder_keys]
    
    # 检查形状匹配
    shape_mismatches = []
    for k in matched_keys:
        if k in state_dict and hasattr(state_dict[k], 'shape'):
            # 获取模型中参数的形状
            model_param = None
            for name, param in model.named_parameters():
                if name == k:
                    model_param = param
                    break
            
            if model_param is not None and state_dict[k].shape != model_param.shape:
                shape_mismatches.append((k, state_dict[k].shape, model_param.shape))
    
    # 编译报告
    report = {
        'total': {
            'pretrained': len(all_keys),
            'model': len(model_keys),
            'matched': len(matched_keys),
            'extra': len(extra_keys),
            'missing': len(missing_keys),
            'shape_mismatches': len(shape_mismatches)
        },
        'encoder': {
            'pretrained': len(encoder_keys),
            'model': len(encoder_model_keys),
            'matched': len(matched_encoder_keys),
            'extra': len(extra_encoder_keys),
            'missing': len(missing_encoder_keys)
        },
        'decoder': {
            'pretrained': len(decoder_keys),
            'model': len(decoder_model_keys),
            'matched': len(matched_decoder_keys),
            'extra': len(extra_decoder_keys),
            'missing': len(missing_decoder_keys)
        },
        'details': {
            'extra_encoder_keys': extra_encoder_keys[:5],  # 只显示前5个
            'missing_encoder_keys': missing_encoder_keys[:5],
            'shape_mismatches': shape_mismatches[:5]
        }
    }
    
    return report

def compare_encoder_stats_before_after(model, state_dict):
    """
    比较encoder参数在加载前后的统计值变化
    
    Args:
        model: 当前模型实例
        state_dict: 预训练模型的状态字典
    
    Returns:
        dict: 包含encoder参数统计的字典
    """
    # 收集当前模型的encoder参数统计（加载前）
    encoder_stats_before = {}
    for name, param in model.named_parameters():
        if name.startswith('encoder'):
            try:
                encoder_stats_before[name] = {
                    'mean': param.data.mean().item(),
                    'std': param.data.std().item(),
                    'min': param.data.min().item(),
                    'max': param.data.max().item()
                }
            except:
                encoder_stats_before[name] = {
                    'mean': float('nan'),
                    'std': float('nan'),
                    'min': float('nan'),
                    'max': float('nan')
                }
    
    # 尝试加载预训练权重（仅用于测试）
    temp_model = copy.deepcopy(model)
    temp_model.load_state_dict(state_dict, strict=False)
    
    # 收集加载后模型的encoder参数统计
    encoder_stats_after = {}
    for name, param in temp_model.named_parameters():
        if name.startswith('encoder'):
            try:
                encoder_stats_after[name] = {
                    'mean': param.data.mean().item(),
                    'std': param.data.std().item(),
                    'min': param.data.min().item(),
                    'max': param.data.max().item()
                }
            except:
                encoder_stats_after[name] = {
                    'mean': float('nan'),
                    'std': float('nan'),
                    'min': float('nan'),
                    'max': float('nan')
                }
    
    # 计算变化量
    changes = {}
    unchanged_count = 0
    changed_count = 0
    
    for name in encoder_stats_before:
        if name in encoder_stats_after:
            before = encoder_stats_before[name]
            after = encoder_stats_after[name]
            
            # 检查参数是否有明显变化（基于均值）
            mean_diff = abs(before['mean'] - after['mean'])
            
            if mean_diff < 1e-6:  # 很小的变化视为未改变
                unchanged_count += 1
                changes[name] = {'changed': False, 'mean_diff': mean_diff}
            else:
                changed_count += 1
                changes[name] = {'changed': True, 'mean_diff': mean_diff}
    
    # 汇总结果
    results = {
        'total_params': len(encoder_stats_before),
        'changed_params': changed_count,
        'unchanged_params': unchanged_count,
        'change_ratio': changed_count / (len(encoder_stats_before) or 1),
        'sample_changes': list(changes.items())[:5]  # 显示前5个参数的变化情况
    }
    
    return results