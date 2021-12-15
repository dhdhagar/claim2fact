# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) 2021 Dhruv Agarwal and authors of arboEL.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
import torch
from tqdm import tqdm, trange
import os
import pickle5 as pickle
from torch.utils.data import DataLoader, TensorDataset

from pytorch_transformers.tokenization_bert import BertTokenizer

from blink.biencoder.zeshel_utils import world_to_id
from blink.common.params import ENT_START_TAG, ENT_END_TAG, ENT_TITLE_TAG


def select_field(data, key1, key2=None):
    if key2 is None:
        return [example[key1] for example in data]
    else:
        return [example[key1][key2] for example in data]


def get_context_representation(
    sample,
    tokenizer,
    max_seq_length,
    mention_key="mention"
):
    mention_tokens = tokenizer.tokenize(sample[mention_key])
    mention_tokens = ["[CLS]"] + mention_tokens[:max_seq_length - 2] + ["[SEP]"]
    input_ids = tokenizer.convert_tokens_to_ids(mention_tokens)
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    assert len(input_ids) == max_seq_length
    return {
        "tokens": mention_tokens,
        "ids": input_ids,
    }

def filter_learnffc_cand_title(candidate_title):
    candidate_title = candidate_title.replace('PolitiFact | ', '')
    candidate_title = candidate_title.replace(' | Snopes.com', '')
    candidate_title = candidate_title.replace('PolitiFact ', '')
    candidate_title = candidate_title.replace(' Snopescom', '')
    return candidate_title

def get_candidate_representation(
    candidate_desc, 
    tokenizer, 
    max_seq_length, 
    candidate_title=None,
    title_tag=ENT_TITLE_TAG,
):
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    cand_tokens = tokenizer.tokenize(candidate_desc)
    if candidate_title is not None:
        candidate_title = filter_learnffc_cand_title(candidate_title)
        title_tokens = tokenizer.tokenize(candidate_title)
        cand_tokens = title_tokens + [title_tag] + cand_tokens

    cand_tokens = cand_tokens[: max_seq_length - 2]
    cand_tokens = [cls_token] + cand_tokens + [sep_token]

    input_ids = tokenizer.convert_tokens_to_ids(cand_tokens)
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    assert len(input_ids) == max_seq_length

    return {
        "tokens": cand_tokens,
        "ids": input_ids,
    }


def process_mention_data(
    samples,
    tokenizer,
    max_context_length,
    max_cand_length,
    silent,
    mention_key="mention",
    context_key="context",
    label_key="label",
    title_key='label_title',
    ent_start_token=ENT_START_TAG,
    ent_end_token=ENT_END_TAG,
    title_token=ENT_TITLE_TAG,
    debug=False,
    logger=None,
    params=None
):
    dict_fpath = os.path.join(params["data_path"], 'dictionary.pickle')
    with open(dict_fpath, 'rb') as read_handle:
        dictionary = pickle.load(read_handle)
    dictionary = {str(fact["cui"]): fact for fact in dictionary}
    
    processed_samples = []

    if debug:
        samples = samples[:200]

    if silent:
        iter_ = samples
    else:
        iter_ = tqdm(samples)

    id_to_idx = {}
    label_id_is_int = False  # Forcing this to be False in order to compute small int labels

    for idx, sample in enumerate(iter_):
        context_tokens = get_context_representation(
            sample,
            tokenizer,
            max_context_length,
            mention_key,
        )

        label = dictionary[str(sample["label_id"])]["description" if not params["use_desc_summaries"] else "summary"]
        title = dictionary[str(sample["label_id"])]["title"]
        label_tokens = get_candidate_representation(
            label, tokenizer, max_cand_length, title,
        )
        
        if label_id_is_int:
            try:
                label_idx = int(sample["label_id"])
            except:
                label_id_is_int = False
        if not label_id_is_int:
            if sample["label_id"] not in id_to_idx:
                id_to_idx[sample["label_id"]] = len(id_to_idx.keys())
            label_idx = id_to_idx[sample["label_id"]]

        record = {
            "context": context_tokens,
            "label": label_tokens,
            "label_idx": [label_idx],
        }

        processed_samples.append(record)

    if logger:
        logger.info("====Processed samples: ====")
        for sample in processed_samples[:5]:
            logger.info("Context tokens : " + " ".join(sample["context"]["tokens"]))
            logger.info(
                "Context ids : " + " ".join([str(v) for v in sample["context"]["ids"]])
            )
            logger.info("Label tokens : " + " ".join(sample["label"]["tokens"]))
            logger.info(
                "Label ids : " + " ".join([str(v) for v in sample["label"]["ids"]])
            )
            logger.info("Label_id : %d" % sample["label_idx"][0])

    context_vecs = torch.tensor(
        select_field(processed_samples, "context", "ids"), dtype=torch.long,
    )
    cand_vecs = torch.tensor(
        select_field(processed_samples, "label", "ids"), dtype=torch.long,
    )
    label_idx = torch.tensor(
        select_field(processed_samples, "label_idx"), dtype=torch.long,
    )
    data = {
        "context_vecs": context_vecs,
        "cand_vecs": cand_vecs,
        "label_idx": label_idx,
    }

    tensor_data = TensorDataset(context_vecs, cand_vecs, label_idx)
    return data, tensor_data
