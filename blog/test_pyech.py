#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : huhu
# @Time     : 2023/3/17 13:49
# @File     : 1.py.py
# @Project  : blog_04
# @objective: 

import pandas as pd

def print_hi():
    all_words = ["衬衫", "羊毛衫", "雪纺衫", "裤子", "高跟鞋", "袜子",
                 "衬衫", "羊毛衫", "衬衫", "衬衫", "衬衫", "袜子", "羊毛衫",
                 "羊毛衫", "雪纺衫", "裤子", "衬衫", "袜子"]
    name = pd.Series(all_words).value_counts().head(10)
    name = name.to_dict()
    name_key = list(name.keys())
    name_value = list(name.values())
    print(name_key)


if __name__ == '__main__':
    print_hi()
