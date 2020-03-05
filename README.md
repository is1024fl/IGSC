Zero-Shot Recognition through Image-Guided Semantic Classification
===


## Pre-Requisites
1. Compatible with Python 3, Pytorch 1.1.0
2. Download datasets for zero shot learning from http://datasets.d2.mpi-inf.mpg.de/xian/xlsa17.zip
3. Prepare aPY, CUB, AWA2, SUN datasets.

## Run the code
### 1. Data Preprocessing:
```
python data_preprocess/train_test_split.py
python apy_image_crop.py
```
Make sure the correct dataset path in the python file.

### 2. run model
```
python train_test.py
```
Edit the train_test.yaml to modify the experiment parameter.

## Model
![](https://i.imgur.com/baftUBd.png)
