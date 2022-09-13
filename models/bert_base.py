# -*- coding: utf-8 -*-
# @Time    : 2021/2/16 15:43
# @Author  : miliang
# @FileName: bert_base.py
# @Software: PyCharm

from pretrain_model_utils.NEZHA.model_NEZHA import NEZHAModel
from transformers import BertModel, ElectraModel
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.pre_model_type == 'NEZHA':
            self.bert = NEZHAModel(config)
        elif config.pre_model_type == 'ELECTRA':
            self.bert = ElectraModel(config)
        elif config.pre_model_type == 'bert':
            self.pre_model = BertModel.from_pretrained(config.pretrain_model_path)
        else:
            raise ValueError('Pre-train Model type must be NEZHA or ELECTRA or bert!')

        self.dropout = torch.nn.Dropout(config.dropout_rate)
        self.hidden2label = nn.Linear(config.bert_hidden_size, config.relation_num)
        self.criterion = nn.CrossEntropyLoss()


    def forward(self, input_ids_list, input_mask_list, segment_ids_list, label_ids_list):
        sequence_out = self.pre_model(input_ids=input_ids_list,
                                      attention_mask=input_mask_list,
                                      token_type_ids=segment_ids_list)[0]
        sequence_out = self.dropout(sequence_out)
        logits = self.hidden2label(sequence_out)
        predict = logits.argmax(-1)
        logits = logits.view(-1, logits.shape[-1])
        label_ids_list = label_ids_list.view(-1)
        loss = self.criterion(logits, label_ids_list)

        return loss, predict


