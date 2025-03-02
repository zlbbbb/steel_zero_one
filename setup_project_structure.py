#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
创建项目基本结构
"""

import os
import sys

def create_directory_structure():
    """创建项目目录结构"""
    
    # 定义要创建的目录
    directories = [
        'data/raw',
        'data/processed',
        'data/external',
        'notebooks',
        'src/data',
        'src/models',
        'src/environment',
        'config',
        'plots',
        'logs'
    ]
    
    # 创建每个目录
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"创建目录: {directory}")
    
    # 创建__init__.py文件以使目录成为Python包
    init_files = [
        'src/__init__.py',
        'src/data/__init__.py',
        'src/models/__init__.py',
        'src/environment/__init__.py'
    ]
    
    for init_file in init_files:
        with open(init_file, 'w') as f:
            f.write("# 此文件使该目录成为Python包\n")
        print(f"创建文件: {init_file}")
        
    print("项目目录结构创建完成!")

if __name__ == "__main__":
    create_directory_structure()