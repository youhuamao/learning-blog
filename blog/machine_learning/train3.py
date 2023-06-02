#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : huhu
# @Time     : 2023/3/15 9:14
# @File     : train.py.py
# @Project  : blog_04
# @objective:
import pandas as pd
from sklearn.model_selection import train_test_split

# 得到评论，normal_file为存放正常评论的文件，spam_file为存放垃圾评论的文件
train_data = pd.read_csv('data/train/train.csv')
test_data = pd.read_csv('data/test/test.csv')

# print (train_data.head(2))

# 将特征划分到 X 中，标签划分到 Y 中
x = train_data.iloc[:, 1:]
y = train_data.iloc[:, 0]

print(y.head(2))

# 对数据集进行随机划分，训练过程暂时没有使用测试数据
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

x_train.head()


# -*-coding:utf-8-*-


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_regression
from sklearn.metrics import classification_report

# stopword.txt 是停用词存储所在的文件
stopword_file = open("data/stopword/stopword.txt", encoding='utf-8', errors='ignore')
# stopword_file = open("data/stopword/stopwords_zh.txt")
stopword_content = stopword_file.read()
stopword_list = stopword_content.splitlines()
stopword_file.close()

# 常见的特征数值计算类，是一个文本特征提取方法。对于每一个训练文本，它只考虑每种词汇在该训练文本中出现的频率。
# count_vect = CountVectorizer(stop_words=stopword_list, token_pattern=r"(?u)\b\w+\b")
count_vect = CountVectorizer(stop_words=stopword_list)
train_count = count_vect.fit_transform(list(x_train['新闻字符串']))

# tf-idf chi特征选择；类似将自然语言转成机器能识别的向量
tfidf_trainformer = TfidfTransformer()
train_tfidf = tfidf_trainformer.fit_transform(train_count)
select = SelectKBest(chi2, k=100)
train_tfidf_chi = select.fit_transform(train_tfidf, y_train)

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
clf = DecisionTreeClassifier()

# 模型训练
clf.fit(train_tfidf, y_train)

# import ant 准确值
print("train accurancy:", clf.score(train_tfidf, y_train))

# 预测值（结果内容是识别的具体值）
train_pre = clf.predict(train_tfidf)

# 输出分类报告（大概就是准确率、召回率）
print('输出分类报告:', classification_report(train_pre, y_train))




import pickle

with open('model/train0.3_1.pickle', 'wb') as fw:
    pickle.dump(clf, fw)

with open('model/train0.3_2.pickle', 'wb') as fw:
    pickle.dump(count_vect, fw)

with open('model/train0.3_3.pickle', 'wb') as fw:
    pickle.dump(tfidf_trainformer, fw)

# 将特征划分到 X 中，标签划分到 Y 中
test_x = test_data.iloc[:, 1:]
test_y = test_data.iloc[:, 0]

# # 读取模型
# with open('model/clf.pickle', 'rb') as clf:
#     clf1 = pickle.load(clf)
#
# with open('model/count_vect.pickle', 'rb') as count_vect:
#     count_vect1 = pickle.load(count_vect)
#
# with open('model/tfidf_trainformer.pickle', 'rb') as tfidf_trainformer:
#     tfidf_trainformer1 = pickle.load(tfidf_trainformer)
#
# # 停用词处理等
# test_count = count_vect1.transform(list(test_x['新闻字符串']))
#
# # 特征选择
# test_tfidf = tfidf_trainformer1.transform(test_count)
# select = SelectKBest(chi2, k=100)
# # test_tfidf_chi = select.transform(test_tfidf)
#
# # 使用模型识别数据
# accurancy = clf1.score(test_tfidf, test_y)
#
# # 识别准确率
# print("accurancy", accurancy)
#
# # 识别结果，类型是numpy.int32（可以使用int()直接转换成int型），后面通过excel来存储
# test_pre = clf1.predict(test_tfidf)
#
# test_pre[:10]
