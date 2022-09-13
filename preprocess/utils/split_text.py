# -*- coding: utf-8 -*-
# @Time    : 2021/2/6 12:13
# @Author  : miliang
# @FileName: split_text.py
# @Software: PyCharm

import re


def split_text(text, maxlen, split_pat=None, greedy=False):
    """文本分片
    将超过长度的文本分片成多段满足最大长度要求的最长连续子文本
    约束条件：1）每个子文本最大长度不超过maxlen；
             2）所有的子文本的合集要能覆盖原始文本。
    Arguments:
        text {str} -- 原始文本
        maxlen {int} -- 最大长度

    Keyword Arguments:
        split_pat {str or re pattern} -- 分割符模式 (default: {SPLIT_PAT})
        greedy {bool} -- 是否选择贪婪模式 (default: {False})
                         贪婪模式：在满足约束条件下，选择子文本最多的分割方式
                         非贪婪模式：在满足约束条件下，选择冗余度最小且交叉最为均匀的分割方式

    Returns:
        tuple -- 返回子文本列表以及每个子文本在原始文本中对应的起始位置列表
    """
    STOPS = (
        '\uFF01'  # Fullwidth exclamation mark
        '\uFF1F'  # Fullwidth question mark
        '\uFF61'  # Halfwidth ideographic full stop
        '\u3002'  # Ideographic full stop
        "，"
    )
    SPLIT_PAT = '([{}]”?)'.format(STOPS)
    split_pat = SPLIT_PAT
    if len(text) <= maxlen:
        return [text], [0]
    segs = re.split(split_pat, text)
    sentences = []
    for i in range(0, len(segs) - 1, 2):
        sentences.append(segs[i] + segs[i + 1])
    if segs[-1]:
        sentences.append(segs[-1])
    n_sentences = len(sentences)
    sent_lens = [len(s) for s in sentences]
    alls = []  # 所有满足约束条件的最长子片段
    for i in range(n_sentences):
        length = 0
        sub = []
        for j in range(i, n_sentences):
            if length + sent_lens[j] <= maxlen or not sub:
                sub.append(j)
                length += sent_lens[j]
            else:
                break
        alls.append(sub)
        if j == n_sentences - 1:
            if sub[-1] != j:
                alls.append(sub[1:] + [j])
            break

    if len(alls) == 1:
        return [text], [0]

    if greedy:  # 贪婪模式返回所有子文本
        sub_texts = [''.join([sentences[i] for i in sub]) for sub in alls]
        starts = [0] + [sum(sent_lens[:i]) for i in range(1, len(alls))]
        return sub_texts, starts
    else:  # 用动态规划求解满足要求的最优子片段集
        DG = {}  # 有向图
        N = len(alls)
        for k in range(N):
            tmplist = list(range(k + 1, min(alls[k][-1] + 1, N)))
            if not tmplist:
                tmplist.append(k + 1)
            DG[k] = tmplist

        routes = {}
        routes[N] = (0, -1)
        for i in range(N - 1, -1, -1):
            templist = []
            for j in DG[i]:
                cross = set(alls[i]) & (set(alls[j]) if j < len(alls) else set())
                w_ij = sum([sent_lens[k] for k in cross]) ** 2  # 第i个节点与第j个节点交叉度
                w_j = routes[j][0]  # 第j个子问题的值
                w_i_ = w_ij + w_j
                templist.append((w_i_, j))
            routes[i] = min(templist)

        sub_texts, starts = [''.join([sentences[i] for i in alls[0]])], [0]
        k = 0
        while True:
            k = routes[k][1]
            sub_texts.append(''.join([sentences[i] for i in alls[k]]))
            starts.append(sum(sent_lens[: alls[k][0]]))
            if k == N - 1:
                break

    return sub_texts, starts


def check_split_right(text, split_text, split_index):
    # 检查首字是否正确
    for index, value in enumerate(split_text):
        assert value[0] == text[split_index[index]]
    # 检查拼接之后是否正确
    temp_text = ""
    for i in split_text:
        temp_text += i
    assert temp_text == text


def split_text_by_sentence(text, max_length):
    split_punctuation = ["。", "！", "!"]
    split_sentences, split_index = [], [0]
    for char_index, char in enumerate(text):
        if char in split_punctuation:
            split_sentences.append(text[split_index[-1]:char_index + 1])
            split_index.append(char_index + 1)

    split_sentences.append(text[split_index[-1]:])

    # 验证一句话不能超过512个字符
    for i in split_sentences:
        assert len(i) <= max_length

    last_sentence, current_sentence = "", ""
    new_split_sentences, new_split_index = [], [0]
    for index, sentence in enumerate(split_sentences):
        current_sentence += sentence
        if len(current_sentence) > max_length:
            current_sentence = sentence
            new_split_sentences.append(last_sentence)
            new_split_index.append(split_index[index])
        last_sentence = current_sentence
    new_split_sentences.append(last_sentence)

    # 验证下切完之后是否正确
    check_split_right(text, new_split_sentences, new_split_index)

    return new_split_sentences, new_split_index


