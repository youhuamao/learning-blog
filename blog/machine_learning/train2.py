#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : huhu
# @Time     : 2023/3/20 9:14
# @File     : train.py.py
# @Project  : blog_04
# @objective:
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# 得到评论，normal_file为存放正常评论的文件，spam_file为存放垃圾评论的文件
train_data = pd.read_csv('data/train/train.csv')
test_data  = pd.read_csv('data/test/test.csv')

# print (train_data.head(2))

# 将特征划分到 X 中，标签划分到 Y 中
x = train_data.iloc[:, 1:]
y = train_data.iloc[:, 0]

y_dict_temp = list(set(y.to_list()))
y_dict_temp.sort()

range(len(y_dict_temp))

y_dict = dict([y_dict_temp[index], index] for index in range(len(y_dict_temp)))

# 字典的键值反转
y_dict_temp = dict(zip(y_dict.values(), y_dict.keys()))

y = [y_dict[index] for index in y]

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Embedding, Flatten, Dropout, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split

# 加载数据集，data 和 labels 分别为训练样本和标签
data = x['新闻字符串'].to_list()
labels = y

# 对文本数据进行分词处理
MAX_NB_WORDS = 10000
MAX_SEQUENCE_LENGTH = 1000
# 文本标记实用类。该类允许使用两种方法向量化一个文本语料库： 将每个文本转化为一个整数序列（每个整数都是词典中标记的索引）；
# 或者将其转化为一个向量，其中每个标记的系数可以是二进制值、词频、TF-IDF权重等。
# num_words: 需要保留的最大词数，基于词频
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(data)
sequences = tokenizer.texts_to_sequences(data)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

# 这个函数将num_samples个文本序列列表 (每个序列为整数列表) 转换成一个 2D Numpy数组，数组形状为 (num_samples, num_timesteps)
# 如果指定了参数 maxlen 的值，则num_timesteps的值取maxlen的值，否则num_timesteps的值等于最长序列的长度。
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

# 将标签进行 one-hot 编码
labels = tf.keras.utils.to_categorical(labels)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# 构建模型
model = Sequential()

embedding_dim = 1000

model.add(Embedding(input_dim=MAX_NB_WORDS,
                    output_dim=embedding_dim,
                    input_length=MAX_SEQUENCE_LENGTH))

model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(labels.shape[1], activation='softmax'))

# 编译模型
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 训练模型
model.fit(X_train, y_train, batch_size=256, epochs=200, validation_data=(X_test, y_test))

# 模型存储
save_path = 'model/mytest.h5'

model.save(save_path)
