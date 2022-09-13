# -*- coding: utf-8 -*-
# @Time    : 2021/2/11 10:19
# @Author  : miliang
# @FileName: train_fine_tune.py
# @Software: PyCharm

from config import Config
from total_utils.DataIterator import DataIterator
from total_utils.train_utils import train

if __name__ == '__main__':
    config = Config()
    config.train_init()
    train_iter = DataIterator(config, config.source_data_path+"train.txt", is_test=False)
    dev_iter = DataIterator(config, config.source_data_path+"dev.txt", is_test=True)
    train(config, train_iter, dev_iter)





