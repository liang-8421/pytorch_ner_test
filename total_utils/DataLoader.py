# -*- coding: utf-8 -*-
# @Time    : 2021/2/6 15:37
# @Author  : miliang
# @FileName: DataLoader.py
# @Software: PyCharm
from config import Config

class InputExample(object):
    """A single training/test example for simple sequence classification."""
    def __init__(self, guid, text, label):
        self.guid = guid
        self.text = text
        self.label = label


def load_data(file_path):
    """读取数据"""
    with open(file_path, 'r', encoding='utf-8') as fr:
        text, text_label = [], []
        lines = []
        for line in fr:
            contents = line.strip()
            word = line.strip().split(' ')[0]
            word_label = line.strip().split(' ')[-1]

            if len(contents) == 0:
                # yield (text, text_label)
                lines.append((text, text_label))
                text, text_label = [], []
            else:
                text.append(word)
                text_label.append(word_label)

    return lines


def create_example(file_path):
    """put data into example """
    example = []
    lines = load_data(file_path)
    file_type = file_path.split('/')[-1].split('.')[0]
    for index, content in enumerate(lines):
        guid = "{0}_{1}".format(file_type, str(index))
        text = content[0]
        label = content[1]
        example.append(InputExample(guid=guid, text=text, label=label))

    return example








