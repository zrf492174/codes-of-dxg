#!/usr/bin/env python
# coding: utf-8


# author:engedaxigua

import streamlit as st

PlotTools = st.Page("PlotTools.py", title="画图小工具", icon=":material/add_circle:")
AnalysisTools = st.Page("omicsTools.py", title="分析小工具", icon=":material/add_circle:")
OtherTools = st.Page("otherTools.py", title="其他工具", icon=":material/add_circle:")
Chatbots = st.Page("Deepseek.py", title="Chatbot", icon=":material/add_circle:")

pg = st.navigation([PlotTools, AnalysisTools,OtherTools,Chatbots])
st.set_page_config(page_title="主页面", page_icon=":material/edit:")
pg.run()