import os
import csv
import json
from collections import defaultdict
from tqdm import tqdm

from pytorch_transformers.tokenization_bert import BertTokenizer

from IPython import embed


PUBTATOR_FILE = '/mnt/nfs/scratch1/rangell/BLINK/tmp/corpus_pubtator.txt'
PRED_MATCHES_FILE = '/mnt/nfs/scratch1/rangell/BLINK/tmp/matches_pred_corpus_pubtator.tsv'
DATA_DIR = '/mnt/nfs/scratch1/rangell/BLINK/data/'
DATASET = 'medmentions'

OUTPUT_DIR = '/mnt/nfs/scratch1/rangell/BLINK/data/{}/taggerOne'.format(DATASET)

# TODO:
# - save documents with transformed mentions

if __name__ == '__main__':

    # get tokenizer
    tokenizer = BertTokenizer(
        '../lerac/coref_entity_linking/models/biobert_v1.1_pubmed/vocab.txt',
        do_lower_case=False
    )

    # get all of the documents
    raw_docs = defaultdict(str)
    with open(PUBTATOR_FILE, 'r') as f:
        for line in f:
            line_split = line.split('|')
            if len(line_split) == 3:
                _text_to_add = ' ' if line_split[1] == 'a' else ''
                _text_to_add += line_split[2].strip()
                raw_docs[line_split[0]] += _text_to_add

    # get all of the mentions and their tfidf candidates in raw form
    print('Reading pred mentions and tfidf candidates...')
    pred_mention_cands = defaultdict(list)
    with open(PRED_MATCHES_FILE, 'r') as f:
        reader = csv.reader(f, delimiter="\t", quotechar='"')
        keys = next(reader)
        for row in tqdm(reader):
            pred_mention_key = (row[0], row[1], row[2])
            pred_mention_cand_val = {k : v for k, v in zip(keys, row)}
            pred_mention_cands[pred_mention_key].append(pred_mention_cand_val)
    print('Done.')
    
    pred_mentions = defaultdict(list)
    for key, value in pred_mention_cands.items():
        pred_mentions[key[0]].append(value[0])

    embed()
    exit()
