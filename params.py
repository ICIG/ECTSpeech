from model.utils import fix_len_compatibility


# data parameters
train_filelist_path = 'fs2_txt/train.txt' #same split as in the FastSpeech2 paper
valid_filelist_path = 'fs2_txt/valid.txt'
test_filelist_path = 'fs2_txt/test.txt'
cmudict_path = 'resources/cmu_dictionary'
add_blank = True
n_feats = 80
n_spks = 1   
spk_emb_dim = 64
n_feats = 80
n_fft = 1024
sample_rate = 22050
hop_length = 256
win_length = 1024
f_min = 0
f_max = 8000

# encoder parameters
n_enc_channels = 192
filter_channels = 768
filter_channels_dp = 256
n_enc_layers = 6
enc_kernel = 3
enc_dropout = 0.1
n_heads = 2
window_size = 4

teacher = True    # true for teacher model, false for consistency tuning

# ECT微调参数
teacher_model_path = '/root/shared-nvme/ECT/ECTSpeech_V9/logs/pre_model/model_2070.pt'  # 预训练教师模型路径
double_every = 10    # 每隔10个epoch更新一次微调阶段
use_fp16 = False    # 是否使用混合精度训练，可能在某些GPU上提高稳定性
ect_initial_stage = 0  # 初始微调阶段
ect_min_lr = 1e-5   # 最小学习率，避免学习率过低，预训练模型用1e-4

# 基于epoch的训练参数
steps_per_epoch = 764  # 项目中一个epoch的迭代次数
epoch_to_stage = 10    # 每10个epoch更新阶段，与double_every保持一致

# 迭代计数相关参数
ect_iter_d = steps_per_epoch * double_every  # 每个阶段的迭代步数，以epoch为单位设置
ect_start_iter = 0  # 初始迭代数，用于恢复训练时的连续性
ect_min_stage_length = 3000  # 每个阶段的最小迭代次数，防止阶段过短

# ECT数值稳定性参数
ect_eps = 1e-8      # 防止除零的小值
ect_ratio_clamp = True  # 是否对r/t比率进行裁剪

# 映射函数参数
ect_k = 8.0         # 映射函数k参数，控制小t区域调整强度
ect_b = 1.0         # 映射函数b参数，控制sigmoid函数的陡峭程度
ect_first_stage_scale = 0.5 # 第一阶段的缩放系数，平滑过渡

# 自适应权重参数
ect_adap_c = 1e-3 # 自适应权重平滑系数，防止在差异极小时权重爆炸

# 梯度稳定性参数
grad_clip_threshold = 1.0  # 梯度裁剪阈值，保持训练稳定
loss_scale = 1.0     # 损失缩放系数，提高数值稳定性

# EMA相关参数 
# 以下参数基于ect-main项目统一调整
ect_ema_decay = 0.9999   # EMA基础衰减率 - 调整为与ect-main一致
ect_ema_halflife_nimg = 10000  # 半衰期图像数量 - 添加与ect-main一致的参数
ect_ema_rampup_ratio = 0.05   # EMA预热比例 - 调整为rampup_ratio与ect-main保持一致

# ECT核心参数
ect_q = 1.5         # 控制ECT一致性比例的q值（越大收敛越快但稳定性下降）
ect_p_mean = -1.2   # P分布均值
ect_p_std = 1.2     # P分布标准差

# 数据参数
sigma_data = 0.5    # 数据分布的标准差
sigma_min = 0.002   # 最小噪声水平
sigma_max = 80      # 最大噪声水平
rho = 7             # EDM噪声调度参数

# training parameters
log_dir = 'logs/Pre_model'  # 日志目录,ect存放为'logs/ect'
test_size = 2
n_epochs = 2300     # 训练总轮数,ect为230
batch_size = 16     # 批量大小
learning_rate = 1e-4  # 学习率，Pre_model学习率1e-4，ect学习率1e-5
seed = 1234
save_every = 1      # 每1个epoch保存一次模型
out_size = fix_len_compatibility(2*22050//256)

# 多步采样参数
diffusion_steps = 50  # 扩散模型步数
