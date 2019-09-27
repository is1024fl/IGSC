import os
from os.path import join as PJ
import scipy.io as sio
from pandas import DataFrame as df
import yaml
from sklearn.preprocessing import LabelBinarizer

# Setting
DATASET = "awa2"

ROOT = PJ("..", "dataset")
XLSA17 = PJ(ROOT, "xlsa17", "data", DATASET)
CONCEPT = PJ(ROOT, DATASET, "list", "concepts", "concepts.txt")
CONCEPT_LIST = PJ(ROOT, DATASET, "list", "concepts")
SPLIT_LIST = {
    "train_test": PJ(ROOT, DATASET, "list", "train_test"),
    "train_val": PJ(ROOT, DATASET, "list", "train_test")
}

ATT_SPLITS = sio.loadmat(PJ(XLSA17, "att_splits.mat"))
RES101 = sio.loadmat(PJ(XLSA17, "res101.mat"))

####################
# check path exist #
####################
if not os.path.isdir(PJ(ROOT, DATASET, "list")):
    os.makedirs(PJ(ROOT, DATASET, "list"))

if not os.path.isdir(CONCEPT_LIST):
    os.makedirs(CONCEPT_LIST)

for sl in SPLIT_LIST:
    if not os.path.isdir(SPLIT_LIST[sl]):
        os.makedirs(SPLIT_LIST[sl])

#########################
# generate concept file #
#########################
with open(CONCEPT, "w") as f:
    f.write('\n'.join([label[0][0] for label in ATT_SPLITS['allclasses_names']]))

################################
# generate attribute embedding #
################################
class2att_dict = {label[0][0]: attr.tolist() for label, attr
                  in zip(ATT_SPLITS['allclasses_names'], ATT_SPLITS['att'].T)}

df.from_dict(class2att_dict).to_csv(PJ(CONCEPT_LIST, "concepts_attr.txt"), index=False)

###################
# reorganize data #
###################
img_files = [filter(None, i[0][0].split('/')) for i in RES101['image_files']]
img_files = [PJ(*list(i)[5:]) for i in img_files]

lb = LabelBinarizer()

labels = RES101['labels'].reshape(-1)
labels = labels - 1

if DATASET == 'apy':
    img_files = [PJ("img", i) for i in sorted(os.listdir(PJ(ROOT, DATASET, "img", "img")))]

data = df({'img_path': img_files, 'label': labels})
binary_data = df(lb.fit_transform(labels).tolist())
binary_data.insert(loc=0, column='img_path', value=img_files)

#######################
# split into trainval #
#######################
trainval_data = binary_data.iloc[ATT_SPLITS['trainval_loc'].reshape(-1) - 1]
for i in range(1, 4):

    with open(CONCEPT) as f:
        concepts = [l.strip() for l in f.readlines()]

    with open(PJ(XLSA17, 'trainclasses' + str(i) + '.txt')) as f:
        train_class_names = [l.strip() for l in f.readlines()]

    with open(PJ(XLSA17, 'valclasses' + str(i) + '.txt')) as f:
        val_class_names = [l.strip() for l in f.readlines()]

    train_id = sorted([concepts.index(t) for t in train_class_names])
    test_id = sorted([concepts.index(v) for v in val_class_names])
    ids = {"train_id": sorted(train_id), "test_id": sorted(test_id)}

    with open(PJ(SPLIT_LIST['train_val'], "id_split" + str(i) + ".txt"), "w") as f:
        yaml.dump(ids, f)

    split_mode = ['train', 'val']
    for sm, tid in zip(split_mode, ids):
        split_data = binary_data[data.label.isin(ids[tid])]
        split_data.to_csv(PJ(SPLIT_LIST['train_val'], sm + str(i) + ".txt"), index=False, header=False)

####################################################
# split into train_val, test_seen, and test_unseen #
####################################################
split_mode = ['trainval', 'test_seen', 'test_unseen']
for sm in split_mode:
    split_data = binary_data.iloc[ATT_SPLITS[sm + '_loc'].reshape(-1) - 1]
    split_data.to_csv(PJ(SPLIT_LIST['train_test'], sm + ".txt"), index=False, header=False)

train_id = list(set(data.iloc[ATT_SPLITS['trainval_loc'].reshape(-1) - 1]['label']))
test_id = list(set(data.iloc[ATT_SPLITS['test_unseen_loc'].reshape(-1) - 1]['label']))

ids = {"train_id": sorted(train_id), "test_id": sorted(test_id)}
with open(PJ(SPLIT_LIST['train_test'], "id_split.txt"), "w") as f:
    yaml.dump(ids, f)
