# -*- coding: utf-8 -*-
# @Time    : 2021/2/4 10:12
# @Author  : miliang
# @FileName: data_preprcesss.py
# @Software: PyCharm
from config import Config
import numpy as np

from preprocess.utils.common import get_train_path_list, get_test_path_list, split_train_dev_index, write_train_dev_source_file, write_test_source_file



if __name__ == '__main__':
    config = Config()
    train_path_list = get_train_path_list(config.train_text_path, config.train_label_path)
    test_path_list = get_test_path_list(config.test_text_path)

    # split train/dev index
    train_list_index, dev_list_index = split_train_dev_index(train_path_list)

    write_train_dev_source_file(train_path_list, train_list_index, config.source_data_path, "train.txt")
    write_train_dev_source_file(train_path_list, dev_list_index, config.source_data_path, "dev.txt")
    write_test_source_file(test_path_list, config.source_data_path, "test.txt")















