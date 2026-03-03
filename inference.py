import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import json
import datetime as dt
import numpy as np
from scipy.io.wavfile import write

import torch
import math
import params
from model.tts import Comospeech
from text import text_to_sequence, cmudict
from text.symbols import symbols
from utils import intersperse

import warnings
# 忽略 torch.load 相关的警告
warnings.filterwarnings("ignore", message=".*torch.load.*weights_only=False.*")
# 忽略 weight_norm 相关的警告
warnings.filterwarnings("ignore", message=".*weight_norm is deprecated.*")

import sys
sys.path.append('./hifi-gan/')
from env import AttrDict
from models import Generator as HiFiGAN


HIFIGAN_CONFIG = './checkpts/hifigan-config.json'
HIFIGAN_CHECKPT = './checkpts/hifigan.pt'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, default='text.txt', help='path to a file with texts to synthesize')
    parser.add_argument('-c', '--checkpoint', type=str, default='', help='path to a checkpoint of CoMoSpeech model')
    parser.add_argument('-t', '--timesteps', type=int, default=1, help='number of sampling timesteps (1 for ECT, 10-20 for teacher)')
    parser.add_argument('--teacher', action='store_true', help='Use teacher model (default: use ECT model)')
    parser.add_argument('--use_ema', action='store_true', default=True, help='Use EMA model weights for ECT inference (default: True, produces better quality)')
    parser.add_argument('--no_ema', action='store_true', help='Disable EMA weights for ECT inference')
    parser.add_argument('-o', '--output_dir', type=str, default='out', help='output directory for generated audio files')
    parser.add_argument('--temperature', type=float, default=1.0, help='noise temperature for sampling (only for teacher model)')
    parser.add_argument('--use_ect_generator', action='store_true', help='使用ECT项目兼容的生成器函数进行推理')
 
    args = parser.parse_args()
    
    # 如果指定了no_ema，则禁用EMA
    if args.no_ema:
        args.use_ema = False
    
    # 确认检查点文件存在
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"模型检查点文件不存在: {args.checkpoint}")
    
    # 确定使用哪种模型类型
    use_teacher = args.teacher
    model_type = "Teacher" if use_teacher else "ECT"
    
    # EMA使用验证
    use_ema = args.use_ema
    if use_teacher and use_ema:
        print("警告: 教师模型不支持EMA权重，将忽略--use_ema参数")
        use_ema = False
    
    # 设置合适的采样步数
    if use_teacher and args.timesteps < 10:
        print(f"警告: 教师模型通常需要更多采样步数. 当前设置为 {args.timesteps} 步.")
    elif not use_teacher and args.timesteps > 2:
        print(f"警告: ECT模型通常只需要1-2步采样. 当前设置为 {args.timesteps} 步，这可能会降低速度而不会显著提高质量.")
    
    # ECT生成器验证
    use_ect_generator = args.use_ect_generator
    if use_teacher and use_ect_generator:
        print("警告: 教师模型不支持ECT生成器，将忽略--use_ect_generator参数")
        use_ect_generator = False
    
    print(f'初始化 {model_type} 模型...')
    
 
    # 创建模型实例
    generator = Comospeech(len(symbols)+1, params.n_spks, params.spk_emb_dim,
                    params.n_enc_channels, params.filter_channels,
                    params.filter_channels_dp, params.n_heads, params.n_enc_layers,
                    params.enc_kernel, params.enc_dropout, params.window_size,
                    params.n_feats, teacher=use_teacher).cuda()
    
    print(f'加载模型检查点: {args.checkpoint}')
    checkpoint = torch.load(args.checkpoint, map_location=lambda loc, storage: loc)
    
    # 确定checkpoint格式并加载（简化版本）
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        # 新格式 - 只加载模型权重和ECT状态
        generator.load_state_dict(checkpoint['model'])
        
        # 恢复ECT状态（如果存在）
        if not use_teacher and 'stage' in checkpoint and 'ratio' in checkpoint:
            generator.decoder.loss_fn.stage = checkpoint['stage']
            generator.decoder.loss_fn.ratio = checkpoint['ratio']
            print(f"从checkpoint加载ECT状态: stage={generator.decoder.stage}, ratio={generator.decoder.ratio:.4f}")
        elif not use_teacher:
            # 找不到stage和ratio时使用默认值
            generator.decoder.loss_fn.stage = 5
            generator.decoder.loss_fn.ratio = 0.96875
            print(f"使用默认ECT状态: stage={generator.decoder.stage}, ratio={generator.decoder.ratio:.4f}")
    else:
        # 旧格式 - 直接加载模型权重
        generator.load_state_dict(checkpoint)
        if not use_teacher:
            # 使用默认ECT状态
            generator.decoder.loss_fn.stage = 5
            generator.decoder.loss_fn.ratio = 0.96875
            print(f"使用默认ECT状态: stage={generator.decoder.stage}, ratio={generator.decoder.ratio:.4f}")
    
    # 简化EMA处理
    if not use_teacher:
        if use_ema and hasattr(generator.decoder, 'denoise_fn_ema'):
            print(f"使用EMA权重进行推理（推荐设置）")
        elif not use_ema:
            print(f"使用主网络权重进行推理")
        else:
            print(f"警告: 模型没有EMA权重，将使用主网络权重")
            use_ema = False
    
    # 诊断EDMPrecond函数
    if not use_teacher and hasattr(generator.decoder, 'EDMPrecond'):
        print("\n检查EDMPrecond函数配置:")
        print(f"sigma_data = {generator.decoder.sigma_data}")
        print(f"sigma_min = {generator.decoder.sigma_min}")
        print(f"sigma_max = {generator.decoder.sigma_max}")
    
    # 生成器方法选择信息
    if not use_teacher:
        if use_ect_generator:
            print(f"使用ECT兼容生成器进行推理 (ECT_generator)")
            if args.timesteps > 1:
                print(f"多步采样: 使用{args.timesteps}个中间步骤")
        else:
            print(f"使用标准CT采样器进行推理 (CT_sampler)")
    
    _ = generator.cuda().eval()
    print(f'模型参数数量: {generator.nparams}')
    
    print('初始化 HiFi-GAN vocoder...')
    with open(HIFIGAN_CONFIG) as f:
        h = AttrDict(json.load(f))
    vocoder = HiFiGAN(h)
    vocoder.load_state_dict(torch.load(HIFIGAN_CHECKPT, map_location=lambda loc, storage: loc)['generator'])
    _ = vocoder.cuda().eval()
    vocoder.remove_weight_norm()
    
    # 读取输入文本
    print(f'读取文本文件: {args.file}')
    with open(args.file, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f.readlines()]
    if len(texts) == 0:
        raise ValueError(f"文本文件为空: {args.file}")
    
    cmu = cmudict.CMUDict('./resources/cmu_dictionary')
    save_dir = args.output_dir
    os.makedirs(save_dir, exist_ok=True)
    
    print(f'开始合成语音...')
    print(f'使用模型: {model_type}{"(EMA)" if use_ema else ""}')
    print(f'采样步数: {args.timesteps}')
    print(f'待合成文本数量: {len(texts)}')
    
    # 用于记录推理数据，最后统一计算RTF
    inference_times = []
    mel_lengths = []
    
    with torch.no_grad():
        for i, text in enumerate(texts):
            print(f'合成第 {i+1}/{len(texts)} 个文本: "{text[:30]}..."', end=' ')
            x = torch.LongTensor(intersperse(text_to_sequence(text, dictionary=cmu), len(symbols))).cuda()[None]
            x_lengths = torch.LongTensor([x.shape[-1]]).cuda()
            
            t = dt.datetime.now()
            # ECT模型使用改进的推理方式
            if not use_teacher:
                # 为ECT模型明确指定use_ema选项和use_ect_generator选项
                y_enc, y_dec, attn = generator.forward(
                    x, 
                    x_lengths, 
                    n_timesteps=args.timesteps, 
                    use_ema=use_ema,
                    use_ect_generator=use_ect_generator,
                    temperature=args.temperature
                )
            else:
                # 教师模型使用原始推理方式
                y_enc, y_dec, attn = generator.forward(
                    x, 
                    x_lengths, 
                    n_timesteps=args.timesteps,
                    temperature=args.temperature
                )
            t = (dt.datetime.now() - t).total_seconds()

            # 只记录数据，不计算RTF
            inference_times.append(t)
            mel_lengths.append(y_dec.shape[-1])
            
            print('Done')
            
            # 使用vocoder生成音频
            audio = (vocoder.forward(y_dec).cpu().squeeze().clamp(-1, 1).numpy() * 32768).astype(np.int16)
            
            # 保存音频文件
            out_path = os.path.join(save_dir, f'sample_{i + 1}.wav')
            write(out_path, 22050, audio)
    # 推理完成后统一计算RTF统计
    if inference_times:
        rtf_values = []
        total_inference_time = 0
        total_audio_duration = 0
        
        for inf_time, mel_len in zip(inference_times, mel_lengths):
            audio_duration = mel_len * 256 / 22050  # mel_frames * hop_length / sample_rate
            rtf = inf_time / audio_duration
            rtf_values.append(rtf)
            total_inference_time += inf_time
            total_audio_duration += audio_duration
        
        print(f'\n=== RTF统计信息 ===')
        print(f'总共合成文本数量: {len(rtf_values)}')
        print(f'总推理时间: {total_inference_time:.4f}s')
        print(f'总音频时长: {total_audio_duration:.4f}s')
        print(f'整体RTF: {total_inference_time/total_audio_duration:.4f} ⭐️ (标准RTF值)')
        print(f'平均RTF: {sum(rtf_values)/len(rtf_values):.4f} (每样本RTF的平均)')
        print(f'最小RTF: {min(rtf_values):.4f}')
        print(f'最大RTF: {max(rtf_values):.4f}')
        print(f'RTF标准差: {np.std(rtf_values):.4f}')
        print(f'')
        print(f'💡 论文比较建议使用整体RTF值: {total_inference_time/total_audio_duration:.4f}')
        
        avg_rtf = sum(rtf_values) / len(rtf_values)
        overall_rtf = total_inference_time / total_audio_duration
        if overall_rtf < 1.0:
            print(f'✅ 整体RTF < 1.0，可以实时生成')
        else:
            print(f'❌ 整体RTF > 1.0，无法实时生成')
        print('===================')
        
    print('Done. Check out `out` folder for samples.') 