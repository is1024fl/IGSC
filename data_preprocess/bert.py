import re
import pandas as pd
import numpy as np
import scipy.io as sio
from os.path import join as PJ

DATASET = "awa2"
concept_vec_filename = "concepts_bert.txt"

ROOT = PJ("..", "dataset")
concept_filename = PJ(ROOT, DATASET, "list", "concepts", "concepts.txt")
concept_vec_filename = PJ(ROOT, DATASET, "list", "concepts", concept_vec_filename)
weight_path = PJ(ROOT, "bert", "bert_word_embedding_all.npy")

ATT_SPLITS = sio.loadmat(PJ(ROOT, "xlsa17", "data", DATASET, "att_splits.mat"))

# Output Replaced Word
if DATASET == 'sun':
    wordlist = [c[0][0] for c in ATT_SPLITS['allclasses_names']]
    with open(concept_filename, "w") as f:
        f.writelines("\n".join(wordlist))
    wordlist = [re.sub("_", " ", word.lower()) for word in wordlist]

elif DATASET == 'cub':
    with open(concept_filename) as f:
        wordlist = [re.sub("_", " ", re.sub("[\d\s\.]", "", word).lower()) for word in f.readlines()]

elif DATASET == 'awa2':
    with open(concept_filename) as f:
        wordlist = [re.sub("\+", " ", re.sub("[\d\s\.]", "", word).lower()) for word in f.readlines()]

else:
    with open(concept_filename) as f:
        wordlist = [re.sub("_", " ", re.sub("[\d\s]", "", word).lower()) for word in f.readlines()]

print(wordlist)


word_embedding = np.load(weight_path)
word_embedding = word_embedding.item()

word_vec_dict = {w: np.array(word_embedding[w].tolist()).reshape(-1) for w in wordlist}
    
pd.DataFrame.from_dict(word_vec_dict).to_csv(concept_vec_filename, index=False)
