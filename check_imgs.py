#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import os

# 检测排除错误图片
check_path = '../../../data/SAR_ship'
bad_names = ['ship02017110301044052']
for name in bad_names:
    if name + '.xml' in os.listdir(check_path + '/Annotations'):
        os.remove(check_path + '/Annotations' + name + '.xml')
        os.remove(check_path + '/JPEGImages' + name + '.jpg')

