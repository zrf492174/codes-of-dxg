#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import requests
from openai import OpenAI

# 设置 DeepSeek 大模型 API 的相关参数
DEEPSEEK_API_URL = "https://api.deepseek.com"
API_KEY = "xxxxx"  # xxxxx替换为你的 DeepSeek API 密钥
client = OpenAI(api_key=API_KEY, base_url=DEEPSEEK_API_URL)
# 初始化聊天历史记录
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 页面标题
st.title("可持续对话聊天机器人")

# 文件上传功能
uploaded_file = st.file_uploader("上传文件", type=["txt", "pdf", "docx"])

# 处理上传的文件
if uploaded_file is not None:
    # 读取文件内容
    file_content = uploaded_file.read().decode("utf-8")
    # 将文件内容添加到聊天历史记录中
    st.session_state.chat_history.append({"role": "user", "content": file_content})

# 用户输入框
user_input = st.text_input("请输入您的问题或消息：")
messages = []

# 发送消息按钮
if st.button("发送"):
    # 将用户输入添加到聊天历史记录中
    st.session_state.chat_history.append({"role": "user", "content": user_input})

#     # 构建请求数据
#     request_data = {
#         "messages": st.session_state.chat_history,
#         "model": "your_preferred_model"  # 替换为你想要使用的 DeepSeek 模型名称
#     }

#     # 设置请求头
#     headers = {
#         "Authorization": f"Bearer {API_KEY}",
#         "Content-Type": "application/json"
#     }

#     # 发送请求到 DeepSeek API
#     response = requests.post(DEEPSEEK_API_URL, json=request_data, headers=headers)
    messages.append({"role": "user", "content":user_input})
    response = client.chat.completions.create(
    model="deepseek-chat",
    messages=st.session_state.chat_history,
    stream=False)
    #messages.append(response.choices[0].message)
    
    
    
    
    # 获取响应数据
    #response_data = response.json()

    # 将机器人的回复添加到聊天历史记录中
    st.session_state.chat_history.append({"role": "assistant", "content": response.choices[0].message.content})

# 显示聊天历史记录
for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.write(f"**用户：** {message['content']}")
    else:
        st.write(f"**机器人：** {message['content']}")
#st.write(st.session_state.chat_history)

