# -*- coding: utf-8 -*-
# @Time    : 2021/2/4 10:14
# @Author  : miliang
# @FileName: common.py
# @Software: PyCharm

import re
import os
import numpy as np
import json
from preprocess.utils.read_data import read_single_train_data,read_single_test_data


def get_sort_file(data_dir):
    """
    Read files sequentially from the folder
    :param data_dir:
    :return:
    """
    file_list = []
    for i in os.listdir(data_dir):
        file_list.append(i)
    file_list.sort(key=lambda x: int(re.findall(r"""(\d+).(.*)""", x, re.DOTALL)[0][0]))
    file_list = list(map(lambda x: data_dir + x, file_list))
    return file_list


def get_train_path_list(data_dir, label_dir):
    """
    read train dataset file
    :param data_dir:
    :param label_dir:
    :return: return a list, evert trupe contain test_file and label_file
    """
    data_file_list = get_sort_file(data_dir)
    label_file_list = get_sort_file(label_dir)
    assert len(data_file_list) == len(label_file_list)

    file_list = []
    for index, _ in enumerate(data_file_list):
        file_list.append((data_file_list[index],label_file_list[index]))
    return file_list


def get_test_path_list(data_dir):
    """
    read test dataset file
    :param data_dir:
    :return:
    """
    file_list = []
    for i in os.listdir(data_dir):
        file_list.append(i)
    file_list.sort(key=lambda x:int(re.findall(r"""(\d+).(.*)""", x, re.DOTALL)[0][0]))
    file_list = list(map(lambda x: data_dir+ x, file_list))
    return file_list


def split_train_dev_index(train_path_list):
    """
    shuffle the index
    :param train_path_list:
    :return:
    """
    train_list_index = list(range(len(train_path_list)))
    np.random.shuffle(train_list_index)
    dev_list_index = train_list_index[:int(len(train_list_index) * 0.2)]
    train_list_index = train_list_index[int(len(train_list_index) * 0.2):]

    return train_list_index, dev_list_index


def write_train_dev_source_file(source_file_list, file_indexs, write_file_path, write_file_name="train.txt"):
    """
    write train/dev source file
    :param source_file_list:
    :param file_indexs:
    :param write_file_path:
    :param write_file_name:
    :return:
    """
    with open(write_file_path+write_file_name, 'w', encoding='utf-8') as fw:
        for file_index in file_indexs:
            texts, labels = read_single_train_data(source_file_list[file_index])
            for text_index, text in enumerate(texts):
                for index, item in enumerate(text):
                    if text[index] == "§" and labels[text_index][index] != "O":
                        print("data process error ！！！")
                    if text[index] != "§":
                        fw.write('{0} {1}\n'.format(text[index], labels[text_index][index]))
                fw.write("\n")


def write_test_source_file(source_file_list, write_file_path, write_file_name="test.txt"):
    test_recoder_json = list()

    with open(write_file_path+write_file_name, 'w', encoding='utf-8') as fw:
        for file_path in source_file_list:
            file_name = file_path.split("/")[-1]
            sub_texts, starts= read_single_test_data(file_path)
            lens = [len(sub_text) for sub_text in sub_texts]
            test_recoder_json.append({"file_name": file_name, "starts": starts, "lens": lens})

            for sub_text in sub_texts:
                sub_text = list(sub_text)
                for c1 in sub_text:
                    if c1 == "\n":
                        raise ValueError(r" Could contain \n")
                        print("存在换行符！！！！！！")
                    fw.write('{0} {1}\n'.format(c1, 'O'))
                fw.write('\n')

    with open(write_file_path + "test_recoder_json.json", "w", encoding="utf-8") as fw:
        json.dump(test_recoder_json, fw, ensure_ascii=False, indent=4)

