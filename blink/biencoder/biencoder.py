# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) 2021 Dhruv Agarwal and authors of arboEL.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from pytorch_transformers.modeling_bert import (
    BertPreTrainedModel,
    BertConfig,
    BertModel,
)

from pytorch_transformers.tokenization_bert import BertTokenizer

from blink.common.ranker_base import BertEncoder, get_model_obj
from blink.common.optimizer import get_bert_optimizer

from collections import OrderedDict
from IPython import embed

def load_biencoder(params):
    # Init model
    biencoder = BiEncoderRanker(params)
    return biencoder


class BiEncoderModule(torch.nn.Module):
    def __init__(self, params):
        super(BiEncoderModule, self).__init__()
        ctxt_bert = BertModel.from_pretrained(params["bert_model"]) # Could be a path containing config.json and pytorch_model.bin; or could be an id shorthand for a model that is loaded in the library
        cand_bert = BertModel.from_pretrained(params["bert_model"])
        self.context_encoder = BertEncoder(
            ctxt_bert,
            params["out_dim"],
            layer_pulled=params["pull_from_layer"],
            add_linear=params["add_linear"],
        )
        self.cand_encoder = BertEncoder(
            cand_bert,
            params["out_dim"],
            layer_pulled=params["pull_from_layer"],
            add_linear=params["add_linear"],
        )
        self.config = ctxt_bert.config

    def forward(
        self,
        token_idx_ctxt,
        segment_idx_ctxt,
        mask_ctxt,
        token_idx_cands,
        segment_idx_cands,
        mask_cands,
    ):
        embedding_ctxt = None
        if token_idx_ctxt is not None:
            embedding_ctxt = self.context_encoder(
                token_idx_ctxt, segment_idx_ctxt, mask_ctxt
            )
        embedding_cands = None
        if token_idx_cands is not None:
            embedding_cands = self.cand_encoder(
                token_idx_cands, segment_idx_cands, mask_cands
            )
        return embedding_ctxt, embedding_cands


class BiEncoderRanker(torch.nn.Module):
    def __init__(self, params, shared=None):
        super(BiEncoderRanker, self).__init__()
        self.params = params
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not params["no_cuda"] else "cpu"
        )
        self.n_gpu = torch.cuda.device_count()
        
        # init tokenizer
        self.NULL_IDX = 0
        self.START_TOKEN = "[CLS]"
        self.END_TOKEN = "[SEP]"
        vocab_path = os.path.join(params["bert_model"], 'vocab.txt')
        if os.path.isfile(vocab_path):
            print(f"Found tokenizer vocabulary at {vocab_path}")
        self.tokenizer = BertTokenizer.from_pretrained(
            vocab_path if os.path.isfile(vocab_path) else params["bert_model"], do_lower_case=params["lowercase"]
        )
        
        # init model
        self.build_model()
        model_path = params.get("path_to_model", None) # Path to pytorch_model.bin for the biencoder model (not the pretrained bert model)
        if model_path is not None:
            self.load_model(model_path)
        self.model = self.model.to(self.device)
        self.data_parallel = params.get("data_parallel")
        if self.data_parallel:
            self.model = torch.nn.DataParallel(self.model)

    def load_model(self, fname, cpu=False):
        if cpu:
            state_dict = torch.load(fname, map_location=lambda storage,location: "cpu")
        else:
            state_dict = torch.load(fname)
        self.model.load_state_dict(state_dict)

    def build_model(self):
        self.model = BiEncoderModule(self.params)

    def save_model(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = get_model_obj(self.model) 
        output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(output_dir, CONFIG_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)

    def get_optimizer(self, optim_states=None, saved_optim_type=None):
        return get_bert_optimizer(
            [self.model],
            self.params["type_optimization"],
            self.params["learning_rate"],
            fp16=self.params.get("fp16"),
        )
 
    def encode_context(self, ctxt, requires_grad=False):
        token_idx_ctxt, segment_idx_ctxt, mask_ctxt = to_bert_input(
            ctxt, self.NULL_IDX
        )
        embedding_context, _ = self.model(
            token_idx_ctxt, segment_idx_ctxt, mask_ctxt, None, None, None
        )
        if requires_grad:
            return embedding_context
        return embedding_context.detach().cpu()

    def encode_candidate(self, cands, requires_grad=False):
        token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(
            cands, self.NULL_IDX
        )
        _, embedding_cands = self.model(
            None, None, None, token_idx_cands, segment_idx_cands, mask_cands
        )
        if requires_grad:
            return embedding_cands
        return embedding_cands.detach().cpu()

    # Score candidates given context input and label input
    # If cand_encs is provided (pre-computed), cand_vecs is ignored
    def score_candidate(
        self,
        text_vecs,
        cand_vecs,
        random_negs=True,
        cand_encs=None,  # pre-computed candidate encoding.
    ):
        # Encode contexts first
        token_idx_ctxt, segment_idx_ctxt, mask_ctxt = to_bert_input(
            text_vecs, self.NULL_IDX
        )
        embedding_ctxt, _ = self.model(
            token_idx_ctxt, segment_idx_ctxt, mask_ctxt, None, None, None
        )

        # Candidate encoding is given, do not need to re-compute
        # Directly return the score of context encoding and candidate encoding
        if cand_encs is not None:
            return embedding_ctxt.mm(cand_encs.t())

        # Train time. We compare with all elements of the batch
        token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(
            cand_vecs, self.NULL_IDX
        )
        _, embedding_cands = self.model(
            None, None, None, token_idx_cands, segment_idx_cands, mask_cands
        )
        if embedding_cands.shape[0] != embedding_ctxt.shape[0]:
            embedding_cands = embedding_cands.view(embedding_ctxt.shape[0], embedding_cands.shape[0]//embedding_ctxt.shape[0], embedding_cands.shape[1]) # batchsize x topk x embed_size

        if random_negs:
            # train on random negatives
            return embedding_ctxt.mm(embedding_cands.t())
        else:
            # train on hard negatives
            embedding_ctxt = embedding_ctxt.unsqueeze(2) # batchsize x embed_size x 1
            scores = torch.bmm(embedding_cands, embedding_ctxt) # batchsize x topk x 1
            scores = torch.squeeze(scores, dim=2) # batchsize x topk
            return scores

    # label_input -- negatives provided
    # If label_input is None, train on in-batch negatives
    def forward(self, context_input, cand_input=None, label_input=None, mst_data=None, pos_neg_loss=False, only_logits=False, rand_negs=False):
        if mst_data is not None:
            if rand_negs:
                context_embeds = self.encode_context(context_input, requires_grad=True)  # batchsize x embed_size
                pos_embeds = mst_data['positive_embeds'] # batchsize x embed_size
                scores = context_embeds.mm(pos_embeds.t())  # batchsize x batchsize
                bs = scores.size(0)
            else:
                context_embeds = self.encode_context(context_input, requires_grad=True).unsqueeze(2) # batchsize x embed_size x 1
                pos_embeds = mst_data['positive_embeds'].unsqueeze(1) # batchsize x 1 x embed_size
                neg_dict_embeds = self.encode_candidate(mst_data['negative_dict_inputs'], requires_grad=True) # (batchsize*knn_dict) x embed_size
                neg_men_embeds = self.encode_context(mst_data['negative_men_inputs'], requires_grad=True) # (batchsize*knn_men) x embed_size
                neg_dict_embeds = neg_dict_embeds.view(context_embeds.shape[0], neg_dict_embeds.shape[0]//context_embeds.shape[0], neg_dict_embeds.shape[1]) # batchsize x knn_dict x embed_size
                neg_men_embeds = neg_men_embeds.view(context_embeds.shape[0], neg_men_embeds.shape[0]//context_embeds.shape[0], neg_men_embeds.shape[1]) # batchsize x knn_men x embed_size
                
                cand_embeds = torch.cat((pos_embeds, neg_dict_embeds, neg_men_embeds), dim=1) # batchsize x knn x embed_size

                # Compute scores
                scores = torch.bmm(cand_embeds, context_embeds) # batchsize x topk x 1
                scores = torch.squeeze(scores, dim=2) # batchsize x topk
        else:
            flag = label_input is None
            scores = self.score_candidate(context_input, cand_input, flag)
            bs = scores.size(0)
        
        if only_logits:
            return scores

        if label_input is None or rand_negs:
            target = torch.LongTensor(torch.arange(bs))
            target = target.to(self.device)
            loss = F.cross_entropy(scores, target, reduction="mean")
        else:
            if not pos_neg_loss:
                loss = torch.mean(torch.max(-torch.log(torch.softmax(scores, dim=1) + 1e-8) * label_input, dim=1)[0])
            else:
                loss = torch.mean(torch.sum(-torch.log(torch.softmax(scores, dim=1) + 1e-8) * label_input - torch.log(1 - torch.softmax(scores, dim=1) + 1e-8) * (1 - label_input), dim=1))
        return loss, scores


def to_bert_input(token_idx, null_idx):
    """ token_idx is a 2D tensor int.
        return token_idx, segment_idx and mask
    """
    segment_idx = token_idx * 0
    mask = token_idx != null_idx
    # nullify elements in case self.NULL_IDX was not 0
    token_idx = token_idx * mask.long()
    return token_idx, segment_idx, mask
