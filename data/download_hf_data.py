"""
==========================
# -*- coding: utf8 -*-
# @Author   : louiss007
# @Time     : 2024/4/3 14:11
# @FileName : download_hf_data.py
# @Email    : quant_master2000@163.com
==========================
"""
from datasets import load_dataset
import pandas as pd


ds = load_dataset('wmt18', 'zh-en')
df = pd.DataFrame(ds['train'])
df.to_csv('data/wmt18/train.csv', index=False)
