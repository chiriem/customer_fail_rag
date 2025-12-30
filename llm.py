from openai import OpenAI
import streamlit as st

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

SYSTEM_PROMPT = """
당신은 실패 사례 분석 전용 데이터 분석가입니다.

제공된 실패 사례 요약만을 근거로 답변해주세요.
정보가 없으면 반드시 '분석 불가'라고 답해주세요.
추측이나 일반론을 사용하지 말아주세요.
"""

def answer_question(question, context_cases):
    context = "\n\n".join(context_cases)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"""
[실패 사례 요약]
{context}

[질문]
{question}
"""
        }
    ]

    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=messages,
        temperature=0.2,
        max_tokens=500
    )

    return response.choices[0].message.content