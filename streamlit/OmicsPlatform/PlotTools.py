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
        ('火山图','小提琴图', '箱线图', '热图', '聚类热图','相关性热图','PCA分析','韦恩图','核密度图')
    )



if chart_type=='火山图':

    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write(data)
    else:
        data = pd.read_csv("./data2test/resultDiff.csv")
        st.write("数据示例")
        st.write(data)

    # 滑动条
    p_value = st.sidebar.text_input("请选择pvalue", '0.05')
    log2FoldChangeValue = st.sidebar.text_input("请选择log2差异倍数", '1')
    AnnotGenes = st.sidebar.text_input("请输入要标注的基因名，基因之间以“,”分隔", '')
    p_value = float(p_value)
    log2FoldChangeValue = float(log2FoldChangeValue)
    if AnnotGenes != '':
        AnnotGenes = list(str(AnnotGenes).split(','))
    else:
        AnnotGenes = []

    data['log10p'] = -1 * np.log10(data['pvalue'])

    data['sig'] = 'unchange'
    data.loc[(data['log2foldchange']> log2FoldChangeValue)&(data['pvalue'] < p_value),'sig'] = 'up'
    data.loc[(data['log2foldchange']< -log2FoldChangeValue)&(data['pvalue'] < p_value),'sig'] = 'down'

    plt.figure(figsize=(11, 9),dpi=300)

    ax = sns.scatterplot(x="log2foldchange", y="log10p",hue='sig',hue_order = ('down','unchange','up'),palette=("#377EB8","#97a9b6","#E41A1C"),data=data)

    if AnnotGenes != []:
        for i in AnnotGenes:
            ii = data.loc[data['Genes']==i].index[0]
            plt.annotate(i,xy = (data.loc[ii,'log2foldchange'],data.loc[ii,'log10p']),xytext=(data.loc[ii,'log2foldchange']+0.05,data.loc[ii,'log10p']+0.05), color='purple')

    ax.set_ylabel('-log(p_value)',fontweight='bold')
    ax.set_xlabel('log2FoldChange',fontweight='bold')
    ax.get_legend().set_title('up/down')

    # 添加水平虚线
    ax.axhline(y=-1 * np.log10(p_value), color='black', linestyle='--')  # y=1 是虚线的位置，颜色为红色，线型为虚线

    # 添加垂直虚线
    ax.axvline(x=log2FoldChangeValue, color='black', linestyle='--')  # x=0 是虚线的位置，颜色为绿色，线型为虚线

    # 添加垂直虚线
    ax.axvline(x=-log2FoldChangeValue, color='black', linestyle='--')  # x=0 是虚线的位置，颜色为绿色，线型为虚线


    st.pyplot(ax.figure)
    plt.show()
elif chart_type=='小提琴图':

    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write(data)
    else:
        data = pd.read_csv("./data2test/data.csv")
        st.write("数据示例")
        st.write(data)

    data2plot = data.iloc[:,2:]
    data2plot = np.log2(data2plot+1)
    data2plot.index = data['Genes']

    ax = sns.violinplot(data=data2plot, palette="Set3")

    ax.set_ylabel('log2 intensity',fontweight='bold')
    ax.set_xlabel('group',fontweight='bold')
    st.pyplot(ax.figure)
elif chart_type=='箱线图':

    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write(data)
    else:
        data = pd.read_csv("./data2test/data.csv")
        st.write("数据示例")
        st.write(data)

    data2plot = data.iloc[:,2:]
    data2plot = np.log2(data2plot+1)
    data2plot.index = data['Genes']

    ax = sns.boxplot(data=data2plot, palette="Set3")

    ax.set_ylabel('log2 intensity',fontweight='bold')
    ax.set_xlabel('group',fontweight='bold')
    st.pyplot(ax.figure)
elif chart_type=='热图':

    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write(data)
    else:
        data = pd.read_csv("./data2test/data.csv")
        st.write("数据示例")
        st.write(data)

    data2plot = data.iloc[:,2:]
    data2plot = np.log2(data2plot+1)
    data2plot.index = data['Genes']

    ax = sns.heatmap(data=data2plot)

    ax.set_ylabel('log2 intensity',fontweight='bold')
    ax.set_xlabel('group',fontweight='bold')  
    st.pyplot(ax.figure)
elif chart_type=='聚类热图':

    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write(data)
    else:
        data = pd.read_csv("./data2test/data.csv")
        st.write("数据示例")
        st.write(data)

    data2plot = data.iloc[:,2:]
    data2plot = np.log2(data2plot+1)
    data2plot.index = data['Genes']
    data2plot = data2plot.fillna(value=0)

    ax = sns.clustermap(data=data2plot)
    st.pyplot(ax.figure)
elif chart_type=='相关性热图':
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write(data)
    else:
        data = pd.read_csv("./data2test/data.csv")
        st.write("数据示例")
        st.write(data)

    vmaxValue = st.sidebar.text_input("请选择vmaxValue", '1')
    vinxValue = st.sidebar.text_input("请选择vmaxValue", '0')
    vmaxValue = float(vmaxValue)
    vinxValue = float(vinxValue)

    data2plot = data.iloc[:,2:]
    data2plot = np.log2(data2plot+1)

    CorrMethod = st.selectbox("选择相关性算法",
        ('pearson','kendall', 'spearman'))

    data2plot = data2plot.corr(method=CorrMethod)

    plt.figure(figsize=(11, 9),dpi=300)
    ax = sns.heatmap(data=data2plot,
            vmax=vmaxValue, 
            vmin=vinxValue,
            #cmap=palettable.cmocean.diverging.Curl_10.mpl_colors,
            cmap='Spectral',
            annot=True,#图中数字文本显示
            fmt=".2f",#格式化输出图中数字，即保留小数位数等
            annot_kws={'size':8,'weight':'normal', 'color':'#253D24'},#数字属性设置，例如字号、磅值、颜色            
           )
    
    st.pyplot(ax.figure)
    plt.show()

elif chart_type=='PCA分析':
    from sklearn.decomposition import PCA
    st.write("数据不能有空缺值，如有，将用1去填补")
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'], key=1)
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write(data)
    else:
        data = pd.read_csv("./data2test/data.csv")
        st.write("数据示例")
        st.write(data)

    st.write("分组信息")

    uploaded_file2 = st.file_uploader("Choose a CSV file", type=['csv'], key=2)
    if uploaded_file2 is not None:
        label = pd.read_csv(uploaded_file2)
        st.write(label)
    else:
        label = pd.read_csv("./data2test/group.csv")
        st.write("数据示例")
        st.write(label)

    data2plot = data.iloc[:,2:]
    data2plot = data2plot.fillna(value=1)
    data2plot = np.log2(data2plot+1)

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data2plot.T)
    pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])

    grouped_data1 = pd.concat([pd.DataFrame(label),pd.DataFrame(pca_result)],axis=1)
    grouped_data1.columns = ['name','group','x','y']
    grouped_data = grouped_data1.groupby('group')

    from matplotlib.patches import Ellipse

    conf = 1.96

    i=0
    for group_name, group_data in grouped_data:
        x = group_data['x']
        y = group_data['y']
        
        # 计算置信椭圆参数
        covariance = np.cov(x, y)
        lambda_, v = np.linalg.eig(covariance)
        lambda_ = conf * np.sqrt(lambda_)  # 乘上置信度系数
        theta = np.degrees(np.arctan2(*v[::-1, 0]))
        
        # 获取散点图颜色，并绘制置信椭圆
        ax = plt.scatter(x, y, label=group_name, marker='o', s=5)
        color = ax.get_facecolors()[0]
        ellipse = Ellipse(xy=(np.mean(x), np.mean(y)), width=lambda_[0] * 2,height=lambda_[1] * 2, angle=theta, edgecolor=color, facecolor='None', lw=1, zorder=1)
        plt.gca().add_patch(ellipse)
        
        
        for i in group_data.index:
            plt.text(group_data['x'][i], group_data['y'][i], group_data['name'][i], fontsize=8, color=color, ha='right', va='bottom')
    plt.legend()
    plt.title('PCA')
    plt.xlabel('PC1:'+str(round(pca.explained_variance_ratio_[0]*100,2))+"%")
    plt.ylabel('PC2:'+str(round(pca.explained_variance_ratio_[1]*100,2))+"%")
    plt.tight_layout()
    st.pyplot(ax.figure)
    plt.show()

elif chart_type=='韦恩图':
    from matplotlib_venn import venn3

    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'], key='venn')
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write(data)
    else:
        data = pd.read_csv("./data2test/Venn.csv")
        st.write("数据示例")
        st.write(data)
    
    subsets = []
    grouplabels = []
    for i in data.columns:
        setCol = set(data[i].drop_duplicates().dropna())
        subsets.append(setCol)
        grouplabels.append(i)
    grouplabels = tuple(grouplabels)


    
    my_dpi=300
    fig = plt.figure(figsize=(800/my_dpi, 600/my_dpi), dpi=my_dpi)
    ax = venn3(
    subsets = subsets,
    set_labels=grouplabels,
    #set_colors=('r','g','b')
    )
    st.pyplot(fig)
    plt.show()

elif chart_type == '核密度图':
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write(data)
    else:
        data = pd.read_csv("./data2test/data.csv")
        st.write("数据示例")
        st.write(data)

    data2plot = data.iloc[:,2:]
    data2plot = np.log2(data2plot+1)
    data2plot.index = data['Genes']

    plt.figure(figsize=(11, 9),dpi=300)
    for i in data2plot.columns:
        ax = sns.kdeplot(data2plot[i],label = i)
    #ax = sns.kdeplot(data2plot,label = data2plot.columns)
    ax.legend(loc="upper right")
    ax.grid()  # 添加网格
    plt.xlabel('Value')
    plt.ylabel("Density")
    st.pyplot(ax.figure)
    plt.show()
