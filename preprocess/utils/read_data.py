# -*- coding: utf-8 -*-
# @Time    : 2021/2/4 10:50
# @Author  : miliang
# @FileName: read_data.py
# @Software: PyCharm
import pandas as pd
from total_utils.common import fenge
from preprocess.utils.data_clean import clear_train_text, clear_test_text
from preprocess.utils.split_text import split_text, split_text_by_sentence

def read_single_train_data(file_path):
    """
    read single data
    :param file_path:
    :return:
    """
    with open(file_path[0], 'r', encoding="utf-8") as f:
        text = ""
        for line in f:
            text += line
    text = clear_train_text(text)
    label_df = pd.read_csv(file_path[1])
    label_dict = dict()
    for label_content in label_df.iterrows():
        assert file_path[0].split("/")[-1].split(".")[0] == str(label_content[1]["ID"])
        if str(label_content[1]["Privacy"]) != text[label_content[1]["Pos_b"]:label_content[1]["Pos_e"] + 1]:
            # 进行数据清洗后,找出错误标签
            fenge()
            print_info = "Category is {}：label content is {}(len:{}) ,but index content is {}(len:{})".format(
                                                                                                   label_content[1]["Category"],
                                                                                                   label_content[1]["Privacy"],
                                                                                                   len(label_content[1]["Privacy"]),
                                                                                                   text[label_content[1]["Pos_b"]:label_content[1]["Pos_e"] + 1],
                                                                                                   len(text[label_content[1]["Pos_b"]:label_content[1]["Pos_e"] + 1]))
            print(print_info)
            fenge()
            continue

        if label_content[1]["Category"] not in label_dict:
            label_dict[label_content[1]["Category"]] = list()

        label_dict[label_content[1]["Category"]].append(
                {"start": int(label_content[1]["Pos_b"]),
                 "end": int(label_content[1]["Pos_e"]),
                 "content": label_content[1]["Privacy"]})

    text_label = ["O"] * len(text)
    for label_type, items in label_dict.items():
        for item in items:
            for label_index in range(item["start"], item["end"] + 1):
                text_label[label_index] = "I-" + label_type
            text_label[item["start"]] = "B-" + label_type

    # dynamic split text
    sub_texts, starts = split_text(text, 340, split_pat=None, greedy=True)

    texts, labels = [], []

    for sub_index, sub_text in enumerate(sub_texts):
        temp_text = list(sub_text)
        temp_label = text_label[starts[sub_index]:starts[sub_index] + len(sub_text)]
        assert len(temp_text) == len(temp_label)
        texts.append(temp_text)
        labels.append(temp_label)

    return texts, labels



def read_single_test_data(file_path):
    with open(file_path, 'r', encoding="utf-8") as fr:
        text = ""
        for line in fr:
            line = line.strip()
            text += line

    text = clear_test_text(text)
    sub_texts, starts = split_text(text, 340, split_pat=None, greedy=False)

    # texts, labels = [], []
    # for sub_index, sub_text in enumerate(sub_texts):
    #     temp_text = list(sub_text)
    #     temp_label = ["O"] * len(temp_text)
    #     texts.append(temp_text)
    #     labels.append(temp_label)

    return sub_texts, starts




if __name__ == '__main__':
    path = ('C:/Users/miliang/Desktop/pytorch_ner_baseline/pytorch_ner_baseline/datasets/origin_data/train/data/0.txt', 'C:/Users/miliang/Desktop/pytorch_ner_baseline/pytorch_ner_baseline/datasets/origin_data/train/label/0.csv')
    read_single_train_data(path)




