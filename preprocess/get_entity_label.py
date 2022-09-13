# -*- coding: utf-8 -*-
# @Time    : 2021/2/6 15:41
# @Author  : miliang
# @FileName: get_entity_label.py
# @Software: PyCharm

from config import Config
from preprocess.utils.common import get_train_path_list
import pandas as pd
from tqdm import tqdm

if __name__ == '__main__':
    config = Config()
    train_path_list = get_train_path_list(config.train_text_path, config.train_label_path)
    entity_type = []
    for _, file_name in tqdm(train_path_list):
        df = pd.read_csv(file_name)
        for _, item in df.iterrows():
            if item["Category"] not in entity_type:
                entity_type.append(item["Category"])
    print(entity_type)
    print(len(entity_type))

