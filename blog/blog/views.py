#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : huhu
# @Time     : 2023/3/15 18:32
# @File     : blog_bp.py.py
# @Project  : project_05
# @objective: 

from flask import Flask, request, render_template, redirect, session, flash
from flask import Blueprint, request, render_template, redirect, url_for, flash, send_file
from blog.utils import is_login
import sqlite3
import datetime

# from jinja2 import Markup
from jinja2.utils import markupsafe
from pyecharts import options as opts
from pyecharts.charts import Bar
from create_db import initialization, connect
from blog_06.create_db2 import connect

# 数据库初始化
initialization()

users = [
    {
        'username':'root',
        'password':'root'
    }
]


blog_bp = Blueprint('blog', __name__, url_prefix='/')


# 首页
@blog_bp.route('/')
@is_login
def index():
    conn = sqlite3.connect('blog.db')
    cursor = conn.cursor()
    # conn, cursor = connect()
    # 查询所有文章
    cursor.execute('SELECT id, title, content, created_at FROM posts ORDER BY created_at DESC')
    posts = cursor.fetchall()
    posts = [{'id': id, 'title': title, 'content': content, 'created_at': created_at} for id, title, content, created_at in posts]

    # 渲染模板
    return render_template('index.html', posts=posts)


# @blog_bp.route('/index_01')
# def index_01():
#     # 渲染模板
#     return send_file(r'../templates/base.html')


@blog_bp.route('/biaoge')
def biaoge():
    conn = sqlite3.connect('blog.db')
    cursor = conn.cursor()

    # 查询所有文章
    cursor.execute('SELECT id, title, content, created_at FROM posts ORDER BY created_at DESC')
    posts = cursor.fetchall()
    posts = [{'id': id, 'title': title, 'content': content, 'created_at': created_at} for id, title, content, created_at in posts]

    # 渲染模板
    return render_template('biaoge.html', posts=posts)

# 按文章标题查询文章内容，后台是按文章 ID 查询的，此页面有编辑文章按钮与删除文章按钮
@blog_bp.route('/<int:id>')
@is_login
def show(id):
    conn = sqlite3.connect('blog.db')
    cursor = conn.cursor()

    # 查询文章
    cursor.execute('SELECT id, title, content, created_at, updated_at FROM posts WHERE id = ?', (id,))
    post = cursor.fetchone()
    if post is None:
        return '404 Not Found'
    post = {'id': post[0], 'title': post[1], 'content': post[2], 'created_at': post[3], 'updated_at': post[4]}

    # 渲染模板
    return render_template('show.html', post=post)

# # 查询
# @blog_bp.route('/<title>' ,methods=['GET', 'POST'])
# @is_login
# def find(title):
#     conn = sqlite3.connect('blog.db')
#     cursor = conn.cursor()
#
#     # 查询文章
#     cursor.execute('SELECT id, title, content, created_at, updated_at FROM posts WHERE title = ?', (title,))
#     post = cursor.fetchone()
#     if post is None:
#         return '404 Not Found'
#     post = {'id': post[0], 'title': post[1], 'content': post[2], 'created_at': post[3], 'updated_at': post[4]}
#
#     # 渲染模板
#     return render_template('show.html', post=post)


# 新增文章页面
@blog_bp.route('/create', methods=['GET', 'POST'])
@is_login
def create():
    if request.method == 'POST':
        # 获取表单提交的数据
        title = request.form['title']
        content = request.form['content']
        created_at = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        updated_at = created_at

        # 插入文章
        conn = sqlite3.connect('blog.db')
        cursor = conn.cursor()
        cursor.execute('INSERT INTO posts (title, content, created_at, updated_at) VALUES (?, ?, ?, ?)', (title, content, created_at, updated_at))
        conn.commit()
        conn.close()

        # 显示成功消息
        flash('文章已添加')
        return redirect(url_for('blog.index'))

    # 渲染模板
    return render_template('create.html')

@blog_bp.route('/chaxun', methods=['GET', 'POST'])
@is_login
def find():
    if request.method == 'POST':
        # 获取表单提交的数据
        title = request.form['title']

        # 查询文章
        conn = sqlite3.connect('blog.db')
        cursor = conn.cursor()
        cursor.execute('SELECT id, title, content, created_at, updated_at FROM posts WHERE title = ?', (title,))
        post = cursor.fetchone()
        if post is None:
            return '404 Not Found'
        # post = {'id': post[0], 'title': post[1], 'content': post[2], 'created_at': post[3], 'updated_at': post[4]}

        id = int(post[0])

        return redirect(url_for('blog.show', id=id))


# 从数据库获取文章，并进行更新、编辑
@blog_bp.route('/<int:id>/edit', methods=['GET', 'POST'])
@is_login
def edit(id):
    conn = sqlite3.connect('blog.db')
    cursor = conn.cursor()

    # 查询文章
    cursor.execute('SELECT id, title, content, created_at, updated_at FROM posts WHERE id = ?', (id,))
    post = cursor.fetchone()
    if post is None:
        return '404 Not Found'

    if request.method == 'POST':
        # 获取表单提交的数据
        title = request.form['title']
        content = request.form['content']
        updated_at = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # 更新文章
        cursor.execute('UPDATE posts SET title = ?, content = ?, updated_at = ? WHERE id = ?', (title, content, updated_at, id))
        conn.commit()
        conn.close()

        # 显示成功消息
        flash('文章已更新')
        return redirect(url_for('blog.show', id=id))

    # 渲染模板
    post = {'id': post[0], 'title': post[1], 'content': post[2], 'created_at': post[3], 'updated_at': post[4]}
    return render_template('edit01.html', post=post)


# 从数据删除文章
@blog_bp.route('/<int:id>/delete', methods=['POST'])
@is_login
def delete(id):
    # 删除文章
    conn = sqlite3.connect('blog.db')
    cursor = conn.cursor()
    cursor.execute('DELETE FROM posts WHERE id = ?', (id,))
    conn.commit()
    conn.close()

    # 显示成功消息
    flash('文章已删除')
    return redirect(url_for('blog.index'))


import pandas as pd


# 词频画图
def bar_base(all_words) -> Bar:
    # all_words = ["衬衫", "羊毛衫", "雪纺衫", "裤子", "高跟鞋", "袜子"
    #              "衬衫", "羊毛衫", "衬衫", "衬衫", "衬衫", "袜子"
    #              "羊毛衫", "羊毛衫", "雪纺衫", "裤子", "衬衫", "袜子"]
    words_counts = pd.Series(all_words).value_counts().head(10)
    words_counts = words_counts.to_dict()
    words_counts_key = list(words_counts.keys())
    words_counts_value = list(words_counts.values())

    c = (
        Bar()
        .add_xaxis(words_counts_key)
        .add_yaxis("统计一", words_counts_value)
        .set_global_opts(title_opts=opts.TitleOpts(title="Bar-基本示例", subtitle="词频分析"))
    )
    return c


import jieba
from wordcloud import WordCloud


# 获取文章分词
def cut_word(text):
    # 中文分词dpi,空格隔开每个词
    text = list(jieba.cut(text))
    print('text:', type(text), text)
    return text


# 词频统计
@blog_bp.route('/word_frequency', methods=['GET', 'POST'])
@is_login
def word_frequency():
    # 读取数据
    conn = sqlite3.connect('blog.db')
    cursor = conn.cursor()

    # 查询所有文章
    cursor.execute('SELECT id, title, content, created_at FROM posts ORDER BY created_at DESC')
    posts = cursor.fetchall()
    contents = [content for id, title, content, created_at in posts]
    print ('contents:', contents)

    # 创建停用词列表
    stopwords = [line.strip() for line in open('machine_learning/data/stopword/stopword.txt', encoding='UTF-8').readlines()]
    stopwords += ' '

    # 接收拆分后的数据
    data_new = []
    for sent in contents:
        data_new += cut_word(sent)
    print('data_new:', data_new)

    data_new_list = list(filter(None, (map(lambda x:x if x not in stopwords else '', data_new))))
    print ('data_new_list:', data_new_list)

    # 生成词云
    wordcloud = WordCloud(font_path="static/font/simkai.ttf", width=850, height=400).generate(' '.join(data_new_list))
    wordcloud.to_file('static/image/wordcloud_test_01.jpg')

    c = bar_base(data_new_list)
    return render_template('word_frequency.html', post=markupsafe.Markup(c.render_embed()))


import pickle
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# 读取存储的 sklearn 模型
def get_model():
    # 读取模型
    with open('machine_learning/model/clf.pickle', 'rb') as clf:
        model = pickle.load(clf)

    with open('machine_learning/model/count_vect.pickle', 'rb') as count_vect:
        count_vect = pickle.load(count_vect)

    with open('machine_learning/model/tfidf_trainformer.pickle', 'rb') as tfidf_trainformer:
        tfidf_trainformer = pickle.load(tfidf_trainformer)

    return model, count_vect, tfidf_trainformer

# 读取存储的伯努利模型
def get_model1():
    # 读取模型
    with open('machine_learning/model/train0.1_1.pickle', 'rb') as clf:
        model = pickle.load(clf)

    with open('machine_learning/model/train0.1_2.pickle', 'rb') as count_vect:
        count_vect = pickle.load(count_vect)

    with open('machine_learning/model/train0.1_3.pickle', 'rb') as tfidf_trainformer:
        tfidf_trainformer = pickle.load(tfidf_trainformer)

    return model, count_vect, tfidf_trainformer

# 读取存储的决策时模型
def get_model2():
    # 读取模型
    with open('machine_learning/model/train3.1.pickle', 'rb') as clf:
        model = pickle.load(clf)

    with open('machine_learning/model/train3.2.pickle', 'rb') as count_vect:
        count_vect = pickle.load(count_vect)

    with open('machine_learning/model/train3.3.pickle', 'rb') as tfidf_trainformer:
        tfidf_trainformer = pickle.load(tfidf_trainformer)

    return model, count_vect, tfidf_trainformer


# 获取分类模型的类别
def get_chinese_classifly():
    code_chinese_dict = {
        'news_story':'民生故事',
        'news_culture':'文化',
        'news_entertainment':'娱乐',
        'news_sports':'体育',
        'news_finance':'财经',
        'news_house':'房产',
        'news_car':'汽车',
        'news_edu':'教育',
        'news_tech':'科技',
        'news_military':'军事',
        'news_travel':'旅游',
        'news_world':'国际',
        'stock':'证券股票',
        'news_agriculture':'农业三农',
        'news_game':'电竞游戏',
    }

    return code_chinese_dict


# 文章分类
@blog_bp.route('/categories', methods=['GET', 'POST'])
@is_login
def categories():
    # 读取数据
    conn = sqlite3.connect('blog.db')
    cursor = conn.cursor()

    # 查询所有文章
    cursor.execute('SELECT id, title, content, created_at FROM posts ORDER BY created_at DESC')
    posts = cursor.fetchall()
    posts = [{'id': id, 'title': title, 'content': content, 'created_at': created_at} for id, title, content, created_at in posts]

    # 获取模型
    model, count_vect, tfidf_trainformer = get_model()
    chinese_dict = get_chinese_classifly()
    for index_num in range(len(posts)):
        # 停用词处理等
        print(posts[index_num]['content'])
        test_count = count_vect.transform([posts[index_num]['content']])

        # 特征选择
        test_tfidf = tfidf_trainformer.transform(test_count)
        select = SelectKBest(chi2, k=100)
        # test_tfidf_chi = select.transform(test_tfidf)

        # 识别结果，类型是numpy.int32（可以使用int()直接转换成int型），后面通过excel来存储
        test_pre = model.predict(test_tfidf)
        posts[index_num].update({'classifly':chinese_dict[test_pre[0]]})

        print(posts[index_num]['classifly'])

    return render_template('categories.html',  posts=posts)

# 文章分类
@blog_bp.route('/categories1', methods=['GET', 'POST'])
@is_login
def categories1():
    # 读取数据
    conn = sqlite3.connect('blog.db')
    cursor = conn.cursor()

    # 查询所有文章
    cursor.execute('SELECT id, title, content, created_at FROM posts ORDER BY created_at DESC')
    posts = cursor.fetchall()
    posts = [{'id': id, 'title': title, 'content': content, 'created_at': created_at} for id, title, content, created_at in posts]

    # 获取模型
    model, count_vect, tfidf_trainformer = get_model1()
    chinese_dict = get_chinese_classifly()
    for index_num in range(len(posts)):
        # 停用词处理等
        print(posts[index_num]['content'])
        test_count = count_vect.transform([posts[index_num]['content']])

        # 特征选择
        test_tfidf = tfidf_trainformer.transform(test_count)
        select = SelectKBest(chi2, k=100)
        # test_tfidf_chi = select.transform(test_tfidf)

        # 识别结果，类型是numpy.int32（可以使用int()直接转换成int型），后面通过excel来存储
        test_pre = model.predict(test_tfidf)
        posts[index_num].update({'classifly':chinese_dict[test_pre[0]]})

        print(posts[index_num]['classifly'])

    return render_template('categories1.html',  posts=posts)

@blog_bp.route('/categories2', methods=['GET', 'POST'])
@is_login
def categories2():
    # 读取数据
    conn = sqlite3.connect('blog.db')
    cursor = conn.cursor()

    # 查询所有文章
    cursor.execute('SELECT id, title, content, created_at FROM posts ORDER BY created_at DESC')
    posts = cursor.fetchall()
    posts = [{'id': id, 'title': title, 'content': content, 'created_at': created_at} for id, title, content, created_at in posts]

    # 获取模型
    model, count_vect, tfidf_trainformer = get_model2()
    chinese_dict = get_chinese_classifly()
    for index_num in range(len(posts)):
        # 停用词处理等
        print(posts[index_num]['content'])
        test_count = count_vect.transform([posts[index_num]['content']])

        # 特征选择
        test_tfidf = tfidf_trainformer.transform(test_count)
        select = SelectKBest(chi2, k=100)
        # test_tfidf_chi = select.transform(test_tfidf)

        # 识别结果，类型是numpy.int32（可以使用int()直接转换成int型），后面通过excel来存储
        test_pre = model.predict(test_tfidf)
        posts[index_num].update({'classifly':chinese_dict[test_pre[0]]})

        print(posts[index_num]['classifly'])

    return render_template('categories2.html',  posts=posts)

import numpy as np
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
from keras.utils import pad_sequences
from keras.models import load_model
@blog_bp.route('/categories_dl', methods=['GET', 'POST'])
@is_login
def categories_dl():
    # 读取数据
    conn = sqlite3.connect('blog.db')
    cursor = conn.cursor()

    # 查询所有文章
    cursor.execute('SELECT id, title, content, created_at FROM posts ORDER BY created_at DESC')
    posts = cursor.fetchall()
    posts = [{'id': id, 'title': title, 'content': content, 'created_at': created_at} for id, title, content, created_at in posts]

    # 获取模型
    y_dict_temp = {
        0: 'news_agriculture',
        1: 'news_car',
        2: 'news_culture',
        3: 'news_edu',
        4: 'news_entertainment',
        5: 'news_finance',
        6: 'news_game',
        7: 'news_house',
        8: 'news_military',
        9: 'news_sports',
        10: 'news_story',
        11: 'news_tech',
        12: 'news_travel',
        13: 'news_world',
        14: 'stock'}
    save_path = r'machine_learning/model/mytest.h5'
    model = load_model(save_path)

    # 对文本数据进行分词处理
    MAX_NB_WORDS = 10000
    MAX_SEQUENCE_LENGTH = 1000
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    chinese_dict = get_chinese_classifly()
    for index_num in range(len(posts)):
        sequences = tokenizer.texts_to_sequences(posts[index_num]['content'])
        pred_data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

        # 识别结果，类型是numpy.int32（可以使用int()直接转换成int型），后面通过excel来存储
        test_pre = model.predict(pred_data)
        test_pre = y_dict_temp[np.argmax(test_pre, axis=1)[0]]
        posts[index_num].update({'classifly':chinese_dict[test_pre]})

        print('深度学习预测：', posts[index_num]['classifly'])

    return render_template('categories_dl.html',  posts=posts)


# 关于页面的介绍
@blog_bp.route('/about', methods=['GET', 'POST'])
@is_login
def about():
    return render_template('about.html')


# 注销用户
@blog_bp.route('/logout/')
def logout():
    # 将session中的用户信息删除;
    session.pop('username')
    flash("注销成功")
    return redirect('/login/')


# 注册用户
@blog_bp.route('/register/', methods=['GET', 'POST'])
def register():
    """
        1), http请求的方法为get方法， 直接返回注册页面;
        2). http请求的方法为post方法，
            - 注册的用户名是否已经存在， 如果存在， 重新注册；
            - 如果不存在， 存储用户名和密码到数据库中；
    """
    if request.method == 'GET':
        return render_template('register.html')
    else:
        # 获取post提交的数据
        username = request.form.get('username')
        password = request.form.get('password')
        for user in users:
            # 注册的用户名是否已经存在， 如果存在， 重新注册；
            if user['username'] == username:
                flash("注册失败: 用户名冲突")
                # session['username'] = username
                return redirect('/register/')
        # 如果不存在， 存储用户名和密码到数据库中；
        else:
            users.append(dict(username=username, password=password))
            flash("用户注册成功, 请登录")
            return redirect('/login/')


# 用户登陆
@blog_bp.route('/login/', methods=['GET', 'POST'])
def login():
    # get直接读取填写的数据
    if request.method == 'GET':
        return render_template('login.html')
    # request.method=='POST
    else:
        # 获取post提交的数据
        username = request.form.get('username')
        password = request.form.get('password')
        for user in users:
            if user['username'] == username and user['password'] == password:
                # 存储用户登录信息; session可以认为时字典对象
                session['username'] = username
                # print(session)
                flash("登录成功")
                return redirect('/')
        else:
            flash("登录失败", category='error')
            return render_template('login.html', errMessages='login fail')


