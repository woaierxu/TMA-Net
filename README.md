# TMA-Net: Text-Guided Mismatch-Aware Learning for Semi-Supervised Medical Image Segmentation


## Introduction
Pytorch implementation of our paper "TMA-Net: Text-Guided Mismatch-Aware Learning for Semi-Supervised Medical Image Segmentation".


## Recommended environment:
Please run the following commands.

```
conda create -n tmanet python=3.8
conda activate tmanet
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
cd Code
pip install -r requirements.txt
```

## Data preparation
You can get Left Atrium (LA) dataset from [UA-MT](https://github.com/yulequan/UA-MT/tree/master/data).

You can get Pancreas-CT  (Pancreas) dataset from [Google Drive](https://drive.google.com/file/d/1gipbEAnziAXIWXZZwqE7y2VnTgMrSvXu/view?usp=drive_link) or [Baidu Netdisk](https://pan.baidu.com/s/15SmPKaqeAtT2WV2FcDR2rQ?pwd=rncl).

You can get Kidney Tumor Segmentation 2019 (KiTS19) dataset from [Google Drive](https://drive.google.com/file/d/1FjzVDcEDNchrxqPs5wzoiOTLRHSpJ1HF/view?usp=drive_link) or [Baidu Netdisk](https://pan.baidu.com/s/15SmPKaqeAtT2WV2FcDR2rQ?pwd=rncl).

The dataset split files are provided in `Code/Datasets/`.

Before training or testing, please prepare the local BiomedCLIP files in `Code/biomed_clip/pth/` and generate the text embeddings:

```
cd Code/biomed_clip
python init_biomedclip_textembed.py
```

The generated embeddings will be saved in `Code/biomed_clip/embeddings/`.

## Train

Please run the LA and KiTS19 training scripts from the `Code` directory:

```
cd Code
export PYTHONPATH=..:$PYTHONPATH
```

For Windows PowerShell, use:

```
$env:PYTHONPATH = "..;$env:PYTHONPATH"
```

### LA Training
```
python LA_tmanet_lab10.py --root_path your_LA_dataset_root_path
python LA_tmanet_lab20.py --root_path your_LA_dataset_root_path
```
If the number of labeled data needs to be adjusted, the **labelnum** parameter needs to be modified. For LA, **labelnum** of 8 represents 10% label data, and **labelnum** of 16 represents 20% label data.

### KiTS19 Training
```
python KiTS19_tmanet_lab10.py --root_path your_KiTS19_dataset_root_path
python KiTS19_tmanet_lab20.py --root_path your_KiTS19_dataset_root_path
```
If the number of labeled data needs to be adjusted, the **labelnum** parameter needs to be modified. For KiTS19, **labelnum** of 19 represents 10% label data, and **labelnum** of 38 represents 20% label data.

### Pancreas Training
Please modify the **data_root** parameter in `Code/pancreas/Pancreas_tmanet_lab10.py` or `Code/pancreas/Pancreas_tmanet_lab20.py`.

If the number of labeled data needs to be adjusted, the **label_percent** parameter needs to be modified. For Pancreas, **label_percent** of 10 represents 10% label data, and **label_percent** of 20 represents 20% label data.

```
cd Code/pancreas
export PYTHONPATH=../..:$PYTHONPATH
python Pancreas_tmanet_lab10.py
python Pancreas_tmanet_lab20.py
```

For Windows PowerShell, use:

```
$env:PYTHONPATH = "../..;$env:PYTHONPATH"
```

## Test

### LA Test
Please run the test script from the `Code` directory and modify `test_paths` in `test_LA.py` if needed.

```
cd Code
export PYTHONPATH=..:$PYTHONPATH
python test_LA.py
```

### KiTS19 Test
Please run the test script from the `Code` directory and modify `TEST_PATHS` in `test_KiTS19.py` if needed.

```
cd Code
export PYTHONPATH=..:$PYTHONPATH
python test_KiTS19.py
```

### Pancreas Test
Please run the test script from the `Code/pancreas` directory and modify `test_path` in `test_Pancreas.py` if needed.

```
cd Code/pancreas
export PYTHONPATH=../..:$PYTHONPATH
python test_Pancreas.py
```
