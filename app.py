import streamlit as st
import pandas as pd
import random

df = pd.read_csv('constitution200.csv', encoding='utf-8')

if 'idx' not in st.session_state:
    st.session_state.idx = None

if st.button('問題を出す'):
    st.session_state.idx = random.randint(0, len(df)-1)

if st.session_state.idx is not None:
    question = df.iloc[st.session_state.idx, 1]
    answer = df.iloc[st.session_state.idx, 2]
    st.write(f"問題: {question}")
    if st.button('答えを見る'):
        st.write(f"答え: {answer}")