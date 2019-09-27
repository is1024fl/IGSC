import re
import pandas as pd
import numpy as np
from os.path import join as PJ
import scipy.io as sio
from gensim.models import KeyedVectors

DATASET = "cub"

concept_vec_filename = "concepts_new.txt"
# concept_vec_filename = "concepts_old.txt"
weight_path = "self-enwiki-gensim-normed-win10.bin"
# weight_path = "enwiki-gensim-normed.bin"

ROOT = PJ("..", "dataset")
concept_filename = PJ(ROOT, DATASET, "list", "concepts", "concepts.txt")
concept_vec_filename = PJ(ROOT, DATASET, "list", "concepts", concept_vec_filename)
weight_path = PJ(ROOT, "enwiki", weight_path)

ATT_SPLITS = sio.loadmat(PJ(ROOT, "xlsa17", "data", DATASET, "att_splits.mat"))
RES101 = sio.loadmat(PJ(ROOT, "xlsa17", "data", DATASET, "res101.mat"))

setting = {
    'apy': {
        'replace_word': {
            'diningtable': 'tables',
            'pottedplant': 'houseplant',
            # 'tvmonitor': 'tv'
            'tvmonitor': 'flat_panel_display'
        }},
    'awa2': {
        'replace_word': {}},
    'sun': {
        'replace_word': {}
    },
    'cub': {
        'replace_word': {
            'chuck_will_widow': 'chuckwillswidow',
            'florida_jay': 'florida_scrub_jay',
            'nelson_sharp_tailed_sparrow': 'nelsons_sparrow',
            'cape_glossy_starling': 'cape_starling',
            'artic_tern': 'arctic_tern',
            'black_and_white_warbler': 'blackandwhite_warbler',
            'american_three_toed_woodpecker': 'american_threetoed_woodpecker'
        }
    }
}

# Output Replaced Word

word2vec = KeyedVectors.load(weight_path, mmap='r')
word2vec.wv.vectors_norm = word2vec.wv.vectors

if DATASET == 'sun':
    wordlist = [c[0][0] for c in ATT_SPLITS['allclasses_names']]
    with open(concept_filename, "w") as f:
        f.writelines("\n".join(wordlist))
elif DATASET == 'cub':
    with open(concept_filename) as f:
        wordlist = [re.sub("[\d\s\.]", "", word).lower() for word in f.readlines()]
else:
    with open(concept_filename) as f:
        wordlist = [re.sub("[\d\s]", "", word).lower() for word in f.readlines()]

word_vec_dict = {}

for i, word in enumerate(wordlist):

    tmp_word = setting[DATASET]['replace_word'][word] if word in setting[DATASET]['replace_word'] else word
    query_vec = [np.array(word2vec[q].tolist()).reshape(-1) for q in [tmp_word, tmp_word + "s", "".join(tmp_word.split("_"))]
                 if q in word2vec.wv.vocab]
    word_vec_dict[word] = query_vec if not query_vec else query_vec[0]

    if not query_vec:
        if DATASET == 'apy':
            print(word)

        if DATASET == 'awa2':
            query_vec = [np.array(word2vec[q].tolist()).reshape(-1) for q in ["".join(tmp_word.split("+")), "_".join(tmp_word.split("+"))]
                         if q in word2vec.wv.vocab]
            word_vec_dict[word] = query_vec if not query_vec else query_vec[0]
            if not query_vec:
                print(word2vec.most_similar(word.split("+"), topn=50))
            continue

        if DATASET == 'cub':
            tws = word.split("_")
            query_vec = [np.array(word2vec[q].tolist()).reshape(-1) for q in [tws[0] + "_".join(tws[1:]), "".join(tws[:-1]) + "s_" + tws[-1]]
                         if q in word2vec.wv.vocab]

            if not query_vec:
                print(word)

            word_vec_dict[word] = query_vec if not query_vec else query_vec[0]
            continue

        if DATASET == 'sun':
            print(word)

pd.DataFrame.from_dict(word_vec_dict).to_csv(concept_vec_filename, index=False)
