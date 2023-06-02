#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : huhu
# @Time     : 2023/3/15 18:31
# @File     : app.py.py
# @Project  : project_05
# @objective: 


from flask import Flask


app = Flask(__name__)
app.secret_key = 'afdsgasg352q4tq4tb.;k&*Yg76$^R%^jio09874buasbsda12432'


from blog.views import blog_bp
app.register_blueprint(blog_bp)


if __name__ == '__main__':
    # host='0.0.0.0',
    app.run(debug=True)
