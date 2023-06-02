#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : huhu
# @Time     : 2023/3/20 17:45
# @File     : utils.py.py
# @Project  : blog_04
# @objective: 

from flask import session, flash, request, redirect
from functools import wraps


def is_login(fun):
    """判断用户是否登录成功"""
    @wraps(fun) # 函数被装饰之后,不会修改原本的帮助文档和属性
    def wrapper(*args,**kwargs):
        """
        判断session是否有用户信息
            如果有,说明登录成功
            如果没,说明登录失败,跳转至登录界面
        :param args:
        :param kwargs:
        :return:
        """
        if session.get('username'):
            return fun(*args,**kwargs)

        else:
            # request.path获取用户请求的url地址后面的路径
            flash('用户需要登录才能访问网页： %s' %(request.path))
            return redirect('/login')
    return wrapper

