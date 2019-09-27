from os.path import join as PJ
import scipy.io as sio
from pandas import DataFrame as df
from pandas import read_csv
import cv2
import os

DATASET = "apy"

ROOT = PJ("..", "dataset")
XLSA17 = PJ(ROOT, "xlsa17", "data", DATASET)

ATT_SPLITS = sio.loadmat(PJ(XLSA17, "att_splits.mat"))
RES101 = sio.loadmat(PJ(XLSA17, "res101.mat"))

ORIGIN_ATTR = read_csv(PJ(XLSA17, "origin_attr.txt"), delimiter=" ", header=None)
origin_data = ORIGIN_ATTR.iloc[:, 0: 6]
# print(origin_data)

concepts = [label[0][0] for label in ATT_SPLITS['allclasses_names']]

# reorganize data
img_files = [filter(None, i[0][0].split('/')) for i in RES101['image_files']]
img_files = [PJ(*list(i)[5:]) for i in img_files]

labels = RES101['labels'].reshape(-1)
labels = labels - 1

data = df({'img_path': img_files, 'label': labels})

if not os.path.isdir(PJ(ROOT, DATASET, "img", "img")):
    os.makedirs(PJ(ROOT, DATASET, "img", "img"))

for (i, d1), (_, d2) in zip(data.iterrows(), origin_data.iterrows()):

    img = cv2.imread(PJ(ROOT, DATASET, "img", d1['img_path']))
    x_min, y_min, x_max, y_max = d2[2: 6].values
    croped_img = img[y_min: y_max, x_min: x_max, :]

    cv2.imwrite(PJ(ROOT, DATASET, "img", "img", str(i).zfill(5) + "_" + d2[0].split('/')[-1].strip(".jpg") + "_" + d2[1] + ".jpg"), croped_img)
