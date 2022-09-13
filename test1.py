# -*- coding: utf-8 -*-
# @Time    : 2021/2/16 16:15
# @Author  : miliang
# @FileName: test1.py
# @Software: PyCharm

from tqdm import tqdm
import time
d = {'loss': 0.2, 'learn': 0.8}
for i in tqdm(range(50), desc='进行中', ncols=90, postfix=d):
# desc设置名称,ncols设置进度条长度.postfix以字典形式传入详细信息
    time.sleep(0.1)
    d["loss"] += 1
    pass