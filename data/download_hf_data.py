"""
==========================
# -*- coding: utf8 -*-
# @Author   : louiss007
# @Time     : 2024/4/3 14:11
# @FileName : download_hf_data.py
# @Email    : quant_master2000@163.com
==========================
"""
from datasets import load_dataset, load_from_disk
import pandas as pd


def load_data_from_hf():
    ds = load_dataset('wmt18', 'zh-en')
    df = pd.DataFrame(ds['train'])
    df.to_csv('data/wmt18/train.csv', index=False)


def load_data_from_local():
    ds = load_dataset('data/wmt18/hf_data')
    print(ds['train'])
    # df = pd.DataFrame(ds['train'])
    # df.to_csv('data/wmt18/train.csv', index=False)


if __name__ == '__main__':
    load_data_from_local()