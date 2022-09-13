# -*- coding: utf-8 -*-
# @Time    : 2021/2/4 10:59
# @Author  : miliang
# @FileName: data_clean.py
# @Software: PyCharm
import re

def clear_train_text(text):
    """
    clean train/dev text
    :param text:
    :return:
    """
    special_chars = ("\n", "\u3000", "\u202a", "\u202c")
    text = replace_special_char(text, special_chars, "§")
    website_chars = (r"""http[a-zA-Z]*://[a-zA-Z0-9;&#!:\.//\?=-]*""", r"""www.[a-zA-Z0-9#-/／\.:]*""", r"""http[a-zA-Z]*//:[a-zA-Z0-9#/!\.]*""")
    text = replace_website_char(text, website_chars, "§")
    return text


def clear_test_text(text):
    """
    clean test text
    :param text:
    :return:
    """
    special_chars = ("\n", "\u3000", "\u202a", "\u202c")
    text = replace_special_char(text, special_chars, "")
    website_chars = (r"""http[a-zA-Z]*://[a-zA-Z0-9;&#!:\.//\?=-]*""", r"""www.[a-zA-Z0-9#-/／\.:]*""", r"""http[a-zA-Z]*//:[a-zA-Z0-9#/!\.]*""")
    text = replace_website_char(text, website_chars, "")
    return text


def replace_special_char(text, special_chars, replace_char=""):
    """
    replace special_chars into  replace_char
    :param text:
    :param special_chars:
    :param replace_char:
    :return:
    """
    for special_char in special_chars:
        text = text.replace(special_char, replace_char)
    return text


def replace_website_char(text, website_chars, replace_char="§"):
    for website_char in website_chars:
        html_labels = list(set(re.findall(website_char, text, re.DOTALL)))
        if html_labels:
            for html_label in html_labels:
                text = text.replace(html_label, replace_char * len(html_label))
    return text
