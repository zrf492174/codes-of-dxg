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
        ('富集分析-ORA','富集分析-GSEA','差异检验')
    )



if chart_type=='差异检验':
    st.write("本工具使用t-test进行差异检验，适用于两组比较，t-test前使用levene检验进行方差齐性检验。")
    st.write("请输入需要进行差异检验的数据。")

    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write(data)
    else:
        data = pd.read_csv("./data2test/data.csv")
        st.write("数据示例")
        st.write(data)

    # 滑动条
    p_value = st.sidebar.text_input("请选择pvalue", '0.05')
    log2FoldChangeValue = st.sidebar.text_input("请选择log2差异倍数", '1')
    p_value = float(p_value)
    log2FoldChangeValue = float(log2FoldChangeValue)

    st.write("请输入分组信息。")

    uploaded_file2 = st.file_uploader("Input group info, choose a CSV file", type=['csv'])
    if uploaded_file2 is not None:
        group = pd.read_csv(uploaded_file2)
        st.write(group)
    else:
        group = pd.read_csv("./data2test/group.csv")
        st.write("分组数据示例")
        st.write(data)

    st.write('Using t-test')

    groupName = group['name'].unique()

    samplesA = group[group['name']==groupName[0]]
    samplesB = group[group['name']==groupName[1]]

    A = list(samplesA['samples'])
    B = list(samplesB['samples'])

    leveneCol = []
    pvalueCol = []
    FCcol = []
    data.index = data.iloc[:,1]
    data = data.iloc[:,2:]

    data = data.dropna(axis=0,thresh=0.7*data.shape[1])
    data = np.log2(data + 1)
    data = data.fillna(1)

    st.cache_data
    def calculatePvalue(data):
        for i in range(0,data.shape[0]):
            p_levene = stats.levene(data[A].iloc[i], data[B].iloc[i])[1]
            leveneCol.append(p_levene)
            if p_levene <0.05:
                p = stats.ttest_ind(data[A].iloc[i], data[B].iloc[i] ,equal_var=False)[1]
            else:
                p = stats.ttest_ind(data[A].iloc[i], data[B].iloc[i] ,equal_var=True)[1]
            pvalueCol.append(p)

            FC = np.mean(data[B].iloc[i]) - np.mean(data[A].iloc[i])
            FCcol.append(FC)
        return leveneCol,pvalueCol,FCcol

    leveneCol,pvalueCol,FCcol = calculatePvalue(data)
    
    
    data['p_levene'] = leveneCol
    data['pvalue'] = pvalueCol
    data['log2foldchange'] = FCcol

    st.write(data)

    col1, col2 = st.columns(2)

    

    fig1 = plt.figure(figsize=(10, 8))
    data2plot = data.loc[(data['pvalue'] < p_value) & (abs(data['log2foldchange']) > log2FoldChangeValue)]
    data2plot = data2plot.iloc[:,0:-3]
    ax = sns.clustermap(data=data2plot)  
    with col1:
        st.pyplot(ax.figure)
        plt.show()

    data['log10p'] = -1 * np.log10(data['pvalue'])

    data['sig'] = 'unchange'
    data.loc[(data['log2foldchange']> log2FoldChangeValue)&(data['pvalue'] < p_value),'sig'] = 'up'
    data.loc[(data['log2foldchange']< -log2FoldChangeValue)&(data['pvalue'] < p_value),'sig'] = 'down'

    data = data.sort_values(by = 'pvalue')    

    fig2 = plt.figure(figsize=(10, 8))
    ax2 = sns.scatterplot(x="log2foldchange", y="log10p",hue='sig',hue_order = ('down','unchange','up'),palette=("#377EB8","#97a9b6","#E41A1C"),data=data)

    for i in range(10):
        plt.text(data['log2foldchange'][i]+0.1, data['log10p'][i]+0.1, data.index[i], color='black', size=8, fontweight='bold')

    ax2.set_ylabel('-log(p_value)',fontweight='bold')
    ax2.set_xlabel('log2FoldChange',fontweight='bold')
    ax2.get_legend().set_title('up/down')

    # 添加水平虚线
    ax2.axhline(y=-1 * np.log10(p_value), color='black', linestyle='--')  # y=1 是虚线的位置，颜色为红色，线型为虚线

    # 添加垂直虚线
    ax2.axvline(x=log2FoldChangeValue, color='black', linestyle='--')  # x=0 是虚线的位置，颜色为绿色，线型为虚线

    # 添加垂直虚线
    ax2.axvline(x=-log2FoldChangeValue, color='black', linestyle='--')  # x=0 是虚线的位置，颜色为绿色，线型为虚线

    with col2:
        st.pyplot(ax2.figure)
        plt.show()

elif chart_type=='富集分析-ORA':
    import pandas as pd
    import gseapy as gp
    import matplotlib.pyplot as plt

    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'], key='venn')
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write(data)
    else:
        data = pd.read_csv("./data2test/Genes.csv")
        #st.write("数据示例")
        #st.write(data)
    
    p_value = st.sidebar.text_input("请选择pvalue", '1')
    with st.sidebar:
        Organism = st.selectbox(
            "选择物种名称",
            ('Human', 'Mouse', 'Yeast', 'Fly', 'Fish', 'Worm'))
    Database = st.sidebar.text_input("请选择数据库", 'KEGG_2021_Human')
    top_num = st.sidebar.text_input("请选择展示的term数目", '10')
    p_value = float(p_value)
    top_num = float(top_num)
    Database = Database.split(",")

    glist = data['Genes'].to_list()
    col1, col2 = st.columns(2)

    enr = gp.enrichr(gene_list=glist,
        gene_sets=['KEGG_2021_Human'],
        organism=Organism, # don't forget to set organism to the one you desired! e.g. Yeast
        outdir='./cache/enrich/',top_term=10,cutoff=p_value, format='pdf')

    st.header("富集结果：")
    st.write(enr.results)


    from gseapy import barplot, dotplot
    # categorical scatterplot
    ax1 = dotplot(enr.results,column="Adjusted P-value",
        x='Gene_set', # set x axis, so you could do a multi-sample/library comparsion
        size=10,top_term=5,figsize=(16,30),title = "KEGG",xticklabels_rot=45, # rotate xtick labels
        show_ring=True, # set to False to revmove outer ring
        marker='o',cutoff = p_value)

    with col1:
        st.header("富集结果可视化展示-气泡图：")
        st.pyplot(ax1.figure)
    
    # categorical scatterplot
    ax2 = barplot(enr.results,column="Adjusted P-value",
        group='Gene_set', # set group, so you could do a multi-sample/library comparsion
        size=10,top_term=5,figsize=(16,30),
        #color=['darkred', 'darkblue'] # set colors for group
        color = {'KEGG_2021_Human': 'salmon'},cutoff = p_value)

    with col2:
        st.header("富集结果可视化展示-条形图：")
        st.pyplot(ax2.figure)
elif chart_type=='富集分析-GSEA':
    import pandas as pd
    import gseapy as gp
    import matplotlib.pyplot as plt

    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'], key='venn')
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write(data)
    else:
        data = pd.read_csv("./data2test/gsea_data.gsea_data.rnk",sep='\t')
        #st.write("数据示例")
        #st.write(data)
        data.columns = ['col1','col2']
        rnk = data.sort_values(by=['col2'],ascending=False)
    
    p_value = st.sidebar.text_input("请选择pvalue", '1')
    with st.sidebar:
        Organism = st.selectbox(
            "选择物种名称",
            ('Human', 'Mouse', 'Yeast', 'Fly', 'Fish', 'Worm'))
    Database = st.sidebar.text_input("请选择数据库", 'KEGG_2021_Human')
    top_num = st.sidebar.text_input("请选择展示的term数目", '10')
    p_value = float(p_value)
    top_num = float(top_num)
    Database = Database.split(",")

    pre_res = gp.prerank(rnk=rnk, # or rnk = rnk,
        gene_sets=Database,
        threads=4,
        min_size=5,
        max_size=1000,
        permutation_num=1000, # reduce number to speed up testing
        outdir='./cache.GSEA', # don't write to disk
        seed=6,
        verbose=True, # see what's going on behind the scenes
        )

    col1, col2 = st.columns(2)

    terms = pre_res.res2d.Term
    ax1 = pre_res.plot(terms=terms[1]) # v1.0.5
    
    with col1:
        st.header("富集结果可视化展示-GSEA：")
        st.pyplot(ax1.figure)

    from gseapy import dotplot
    # to save your figure, make sure that ``ofname`` is not None
    ax2 = dotplot(pre_res.res2d,
                column="FDR q-val",
                title='GSEA dotplot',
                cmap=plt.cm.viridis,
                size=6, # adjust dot size
                figsize=(16,30), cutoff=p_value, show_ring=False)

    with col2:
        st.header("富集结果可视化展示-气泡图：")
        st.pyplot(ax2.figure)
    
    st.header('GSEA富集结果')
    st.write(pre_res.res2d)