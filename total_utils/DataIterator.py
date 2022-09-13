# -*- coding: utf-8 -*-
# @Time    : 2021/2/6 15:37
# @Author  : miliang
# @FileName: DataIterator.py
# @Software: PyCharm
from total_utils.DataLoader import create_example
import numpy as np
import torch


class DataIterator(object):
    """
    data iterator
    """
    def __init__(self, config, data_file, is_test=False):
        self.data = create_example(data_file)
        self.batch_size = config.batch_size
        self.device = config.device
        self.sequence_length = config.sequence_length
        self.tokenizer = config.tokenizer
        self.label_map = config.label2id
        self.is_test = is_test
        self.num_records = len(self.data)
        self.all_idx = list(range(self.num_records))  # data index
        self.id_count = 0  # index

        if not self.is_test:
            self.shuffle()

    def convert_single_example(self, example_idx):
        text_list = self.data[example_idx].text
        label_list = self.data[example_idx].label

        if len(text_list) > self.sequence_length - 2:
            text_list = text_list[:(self.sequence_length - 2)]
            label_list = label_list[:(self.sequence_length - 2)]

        char_list, segment_ids, label_ids = [], [], []
        for index, char in enumerate(text_list):
            temp_char = self.tokenizer.tokenize(char.lower())
            if temp_char:
                char_list.append(temp_char[0])
            else:
                # print("unknown character{},use § replace !!!".format(char))
                char_list.append("§")

            segment_ids.append(0)
            label_ids.append(self.label_map[label_list[index]])

        char_list = ["[CLS]"] + char_list + ["[SEP]"]
        segment_ids = [0] + segment_ids + [0]
        label_ids = [self.label_map["[CLS]"]] + label_ids + [self.label_map["[SEP]"]]
        input_ids = self.tokenizer.convert_tokens_to_ids(char_list)
        input_mask = [1] * len(input_ids)

        while len(input_ids) < self.sequence_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(self.label_map["[PAD]"])
            char_list.append("*NULL*")


        assert len(input_ids) == self.sequence_length
        assert len(input_mask) == self.sequence_length
        assert len(segment_ids) == self.sequence_length
        assert len(label_ids) == self.sequence_length


        return input_ids, input_mask, segment_ids, label_ids, char_list


    def shuffle(self):
        np.random.shuffle(self.all_idx)

    def __iter__(self):
        return self

    def __next__(self):
        if self.id_count >= self.num_records:  # Stop iteration condition
            self.id_count = 0
            raise StopIteration

        input_ids_list, input_mask_list, segment_ids_list, label_ids_list, char_lists = [], [], [], [], []
        batch_count = 0
        while batch_count < self.batch_size:
            idx = self.all_idx[self.id_count]
            input_ids, input_mask, segment_ids, label_ids, char_list = self.convert_single_example(idx)

            input_ids_list.append(input_ids)
            input_mask_list.append(input_mask)
            segment_ids_list.append(segment_ids)
            label_ids_list.append(label_ids)
            char_lists.append(char_list)
            batch_count += 1
            self.id_count += 1

            if self.id_count >= self.num_records:
                break


        input_ids_list = torch.tensor([i for i in input_ids_list], dtype=torch.long).to(self.device)
        input_mask_list = torch.tensor([i for i in input_mask_list], dtype=torch.long).to(self.device)
        segment_ids_list = torch.tensor([i for i in segment_ids_list], dtype=torch.long).to(self.device)
        label_ids_list = torch.tensor([i for i in label_ids_list], dtype=torch.long).to(self.device)

        return input_ids_list, input_mask_list, segment_ids_list, label_ids_list, char_lists


    def __len__(self):

        if len(self.data) % self.batch_size == 0:
            return len(self.data) // self.batch_size
        else:
            return len(self.data) // self.batch_size + 1





if __name__ == '__main__':
    from config import Config
    config = Config()
    train_iter = DataIterator(config=config,
                              data_file=config.source_data_path+"train.txt",
                              is_test=False)
    print(len(train_iter))
    num =0
    for input_ids_list, input_mask_list, segment_ids_list, label_ids_list, char_lists in train_iter:
        # print(input_ids_list)
        # print(input_mask_list)
        # print(segment_ids_list)
        # print(label_ids_list)
        # print(char_lists)

        # print(input_ids_list[0])
        # print(label_ids_list[1])
        # break
        print(num)
        num += 1
    # print(len(train_iter))
