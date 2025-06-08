#genai.configure(api_key="AIzaSyBaNoenMBY0EfTcwR_DIPKNgeAdjjh7G90")
import google.generativeai as genai
import streamlit as st
import os

genai.configure(api_key=os.getenv("GEMINI_API_KEY", st.secrets["GEMINI_API_KEY"]))

def get_gemini_summary(prompt_text):
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        chat = model.start_chat()
        resp = chat.send_message(prompt_text)
        return resp.text
    except Exception as e:
        st.error(f"LLM Error: {e}")
        return ""
#AIzaSyBaNoenMBY0EfTcwR_DIPKNgeAdjjh7G90
