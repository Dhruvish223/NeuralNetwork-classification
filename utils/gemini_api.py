import google.generativeai as genai
import streamlit as st

genai.configure(api_key="AIzaSyBaNoenMBY0EfTcwR_DIPKNgeAdjjh7G90")

def get_gemini_summary(prompt_text, role="You are a helpful data science assistant."):
    """
    Universal Gemini summary generator that works with various prompt types.

    Parameters:
    - prompt_text (str): Core user prompt or context
    - role (str): Optional role or instruction to guide Gemini

    Returns:
    - str: Gemini-generated response
    """
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")  # Use gemini-pro (free-tier)
        chat = model.start_chat()
        prompt = f"{role}\n\n{prompt_text}"
        response = chat.send_message(prompt)
        return response.text

    except Exception as e:
        st.error(f"❌ Gemini API Error: {e}")
        return "An error occurred while generating the response."

#gemini-2.0-flash
