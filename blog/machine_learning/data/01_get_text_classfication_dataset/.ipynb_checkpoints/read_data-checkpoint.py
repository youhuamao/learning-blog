#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : huhu
# @Time     : 2023/3/20 9:39
# @File     : read_data.py.py
# @Project  : blog_04
# @objective: 

import pandas as pd

toutiao_data = pd.read_csv('toutiao_cat_data/toutiao_cat_data.txt',
                           names=['新闻ID', '分类code', '分类名称', '新闻字符串', '新闻关键词'],
                           header=None,
                           sep="_!_",
                           )

# print (toutiao_data.head())
toutiao_data.sample(2000)[['分类名称', '新闻字符串']].to_csv('../train/train.csv', index=False)
toutiao_data.sample(1000)[['分类名称', '新闻字符串']].to_csv('../test/test.csv', index=False)
toutiao_data.sample(100)[['分类名称', '新闻字符串']].to_csv('../pred/pred.csv', index=False)


#
# if __name__ == '__main__':
#     print_hi('Python')
