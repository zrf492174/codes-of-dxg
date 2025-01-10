#!/usr/bin/env python
# coding: utf-8


# author:engedaxigua

import pandas as pd
import numpy as np
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
from scipy import stats
import palettable

chart_type = st.selectbox(
        "选择图表类型",
        ("酶切位点标注")
    )



if chart_type=='酶切位点标注':
    cutPoints = st.sidebar.text_input("请选择酶切位点，用“，”分隔", 'K,R')
    rules = st.sidebar.text_input("请选择酶切规则，用“，”分隔", 'P')
    cutPoints = cutPoints.split(',')
    rules = rules.split(',')
    if rules is None:
        rules = ['a']
    else:
        rules = rules

    uploaded_file = st.file_uploader("Choose a txt file，only including sequence of protein", type=['txt'], key=1)
    if uploaded_file is not None:
        #sequenceFile = open(uploaded_file,'r',encoding='utf-8')
        sequence = uploaded_file.read().decode('utf-8')
        #sequence = str(sequence)
        #st.write(sequence)
    else:
        sequence = None

    def color_code_protein_sequence(sequence):
        # 将序列拆分为单个氨基酸
        amino_acids = list(sequence)
        
        # 初始化一个空列表来存储带颜色的氨基酸
        colored_amino_acids = []
        
        # 遍历每个氨基酸及其索引
        for i, aa in enumerate(amino_acids):
            # 检查当前氨基酸是否为K或R
            #if aa == 'T' or aa == 'S' or aa == 'A'or aa == 'V':
            if aa in cutPoints:
                # 检查前一个氨基酸是否为P
                #if i > 0 and amino_acids[i-1] == 'P':
                if i > 0 and amino_acids[i-1] in rules:
                    # 如果是，则标为黄色
                    colored_aa = f"<span style='color: yellow'>{aa}</span>"
                else:
                    # 如果不是，则标为红色
                    colored_aa = f"<span style='color: red'>{aa}</span>"
            # 如果当前氨基酸不是K或R，则保持原样
            else:
                colored_aa = aa
            
            # 将带颜色的氨基酸添加到列表中
            colored_amino_acids.append(colored_aa)
        
        # 将带颜色的氨基酸列表转换回字符串
        colored_sequence = ''.join(colored_amino_acids)
        return colored_sequence

    # 示例蛋白序列
    #protein_sequence = st.text_input("蛋白序列")
    protein_sequence = sequence
    if protein_sequence is not None:
        protein_sequence = protein_sequence
    else:
        protein_sequence = "ACDEFGKPKLRMNPQRSTVWY"
    
    colored_sequence = color_code_protein_sequence(protein_sequence)

    html_code = f"""
    <div style='white-space: pre-wrap; word-wrap: break-word; width: 100%; font-family: monospace;'>
    {colored_sequence}
    </div>
    """
    st.components.v1.html(html_code,height = 600,scrolling = True)
