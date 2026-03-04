下面是完整内容，已加入数据集网址，并说明将数据集放在 `ECTSpeech` 目录下并重命名为 `DUMMY`。

````markdown
# ECTSpeech

## Prepare

Create and activate the conda environment:

```bash
conda create -n Ectspeech python=3.8
conda activate Ectspeech
````

Install PyTorch with CUDA 12.1 support:

```bash
pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 torchaudio==2.4.1+cu121 \
  --index-url https://download.pytorch.org/whl/cu121
```

Install the required dependencies:

```bash
pip install Cython==0.29.23 numpy==1.23.5 matplotlib==3.3.3 einops==0.3.0 \
  inflect==5.0.2 Unidecode==1.1.2 librosa==0.9.2 scipy==1.9.0 tqdm
```

Install additional packages:

```bash
pip install timm==1.0.15
pip install tensorboard
```

## Dataset

We use the LJSpeech dataset for training and tuning.

Download the dataset from:
[https://keithito.com/LJ-Speech-Dataset/](https://keithito.com/LJ-Speech-Dataset/)

After downloading, place the dataset folder under the `ECTSpeech` root directory and rename it to `DUMMY`.

Please prepare the dataset before running the training scripts.

## Pretraining

Run the following command to train the model from scratch:

```bash
python train.py
```

## Tuning

Run the following command to tune the model:

```bash
python tuning.py
```

## Inference

Run the inference script by providing the path to the text file, the checkpoint path, and the number of sampling steps:

```bash
python inference.py -f <text file> -c <checkpoint> -t <sampling steps>
```


