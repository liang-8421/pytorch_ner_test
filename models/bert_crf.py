# -*- coding: utf-8 -*-
# @Time    : 2021/2/16 15:43
# @Author  : miliang
# @FileName: bert_base.py
# @Software: PyCharm

from pretrain_model_utils.NEZHA.model_NEZHA import NEZHAModel
from transformers import BertModel, ElectraModel
from models.layers.crf_layer import CRFLayer
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.pre_model_type == 'NEZHA':
            self.pre_model = NEZHAModel(config)
        elif config.pre_model_type == 'ELECTRA':
            self.pre_model = ElectraModel(config)
        elif config.pre_model_type == 'bert':
            self.pre_model = BertModel.from_pretrained(config.pretrain_model_path)
        else:
            raise ValueError('Pre-train Model type must be NEZHA or ELECTRA or bert!')

        # 参数传递
        self.batch_size = config.batch_size

        self.dropout = torch.nn.Dropout(config.dropout_rate)
        self.hidden2label = nn.Linear(config.bert_hidden_size, config.relation_num)
        self.crf_layer = CRFLayer(config.relation_num, config.label2id)


    def forward(self, input_ids_list, input_mask_list, segment_ids_list, label_ids_list):
        sequence_out = self.pre_model(input_ids=input_ids_list,
                                      attention_mask=input_mask_list,
                                      token_type_ids=segment_ids_list)[0]
        sequence_out = self.dropout(sequence_out)
        logits = self.hidden2label(sequence_out)

        forward_score = self.crf_layer.forward(logits.transpose(1, 0), input_mask_list.transpose(1, 0))
        gold_score = self.crf_layer.score_sentence(logits.transpose(1, 0), label_ids_list.transpose(1, 0), input_mask_list.transpose(1, 0))
        loss = (forward_score - gold_score).sum() / self.batch_size
        predict = self.crf_layer.viterbi_decode(logits.transpose(1, 0), input_mask_list.transpose(1, 0))
        return loss, predict




