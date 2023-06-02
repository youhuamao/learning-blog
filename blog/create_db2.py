#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : huhu
# @Time     : 2023/3/15 18:30
# @File     : create_db.py
# @Project  : project_05
# @objective: 

import mysql.connector
# 初始化数据库
def initialization():
    mydb = mysql.connector.connect(
        host='localhost',
        user='root',
        passwd='g123698745',
        autocommit=True,  # 自动提交，避免修改失败
    )
    mycursor = mydb.cursor()
    mycursor.execute('create database if not exists blogs')


# 创建与blog的链接,cursor对象依赖于mydb对象
def connect():
    conn = mysql.connector.connect(
        host='localhost',
        user='root',
        passwd='g123698745',
        autocommit=True,  # 自动提交，避免修改失败
        database='blogs'
    )
    cursor = conn.cursor()
    cursor.execute(
        '''CREATE TABLE IF NOT EXISTS posts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )'''
    )
    return conn, cursor


# if __name__ == '__main__':
#     print_hi('Python')
