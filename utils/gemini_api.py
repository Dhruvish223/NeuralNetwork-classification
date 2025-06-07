import google.generativeai as genai

genai.configure(api_key="AIzaSyBaNoenMBY0EfTcwR_DIPKNgeAdjjh7G90")

def get_gemini_summary(report_text):
    prompt = f"""You are a data science mentor.
    Analyze the classification report and give:
    - A summary
    - Weak areas
    - Suggestions for improvement
    - Next steps

    Report:
    {report_text}
    """
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt)
    return response.text
