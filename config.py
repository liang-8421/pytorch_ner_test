# -*- coding: utf-8 -*-
# @Time    : 2021/2/4 10:10
# @Author  : miliang
# @FileName: config.py
# @Software: PyCharm
from transformers import BertTokenizer
import torch
import numpy as np
import datetime
import os
from total_utils.common import get_logger


class Config(object):
    def __init__(self):

        # control parameters
        # self.pre_model_type = 'RoBERTa'
        self.pre_model_type = 'bert'
        self.use_multi_gpu = False
        self.device_id = 0
        self.epoch = 20


        #base parameters
        self.batch_size = 50
        self.sequence_length = 256
        self.bert_hidden_size = 768
        self.bert_learning_rate = 1e-4
        self.crf_learning_rate = 1e-4
        self.dropout_rate = 0.2

        self.warmup_prop = 0.1
        self.clip_grad = 2.0
        self.random_seed = 1996

        # train device selection
        if self.use_multi_gpu:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.n_gpu = torch.cuda.device_count()
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            torch.cuda.set_device(self.device_id)
            print('current device:', torch.cuda.current_device())  # watch for current device
            n_gpu = 1
            self.n_gpu = n_gpu


        # set data path
        self.origin_data_path = "/home/data_t/LiangZ/02_code/pytorch_ner_baseline/datasets/origin_data/"
        self.source_data_path = "/home/data_t/LiangZ/02_code/pytorch_ner_baseline/datasets/source_data/"
        self.train_text_path = self.origin_data_path + "train/data/"
        self.train_label_path = self.origin_data_path + "train/label/"
        self.test_text_path = self.origin_data_path + "test/"
        self.model_save_path = "/home/data_t/LiangZ/02_code/pytorch_ner_baseline/model_save/"
        self.config_file_path = "/home/data_t/LiangZ/02_code/pytorch_ner_baseline/config.py"


        # set ner label
        self.entity_label = ['position', 'name', 'movie', 'organization', 'company', 'book', 'address', 'scene', 'mobile', 'email', 'game', 'government', 'QQ', 'vx']
        self.special_label = ["[PAD]", "[CLS]", "[SEP]", "O"]
        self.bio_label = ["B", "I"]
        self.label_type = self.special_label + [j + "-" + i for i in self.entity_label for j in self.bio_label]
        self.label2id = {value: index for index, value in enumerate(self.label_type)}
        self.relation_num = len(self.label_type)


        # 预训练模型存储位置
        # self.pretrain_model_path = "/home/data_t/LiangZ/01_pretrain_model_torch/bert"
        self.pretrain_model_path = "/home/data_t/LiangZ/01_pretrain_model_torch/clue_roberta_chinese_3L768_clue_tiny"
        self.tokenizer = BertTokenizer(vocab_file=self.pretrain_model_path +"/vocab.txt", do_lower_case=True)


    def train_init(self):

        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.cuda.manual_seed_all(self.random_seed)
        torch.backends.cudnn.deterministic = True  # 保证每次结果一样
        self.get_save_path()

    def get_save_path(self):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
        # if self.use_adamw == True:
        #     self.model_save_path = self.model_save_path + self.pre_model_type + "_use_adamw" + "_" + timestamp
        # else:

        self.model_save_path = self.model_save_path + self.pre_model_type + "_" + timestamp

        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

        # 将config文件写入文件夹中
        with open(self.model_save_path + "/config.txt", "w", encoding="utf8") as fw:
            with open(self.config_file_path, "r", encoding="utf8") as fr:
                content = fr.read()
                fw.write(content)

        self.logger = get_logger(self.model_save_path + "/log.log")
        self.logger.info('current device:{}'.format(torch.cuda.current_device()))  # watch for current device


if __name__ == '__main__':
    config = Config()
    print(config.tokenizer.tokenize("\u3000"))



