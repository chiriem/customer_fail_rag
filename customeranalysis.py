import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

import faiss
from openai import OpenAI

# -------------------------
# 기본 설정
# -------------------------
st.set_page_config(page_title="Failure Case Analysis with LLM", layout="wide")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# -------------------------
# 데이터 로드
# -------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("customerdata.csv")

    # 결측치 처리
    df["Work_Experience"] = df["Work_Experience"].fillna(0)
    df["Family_Size"] = df["Family_Size"].fillna(df["Family_Size"].median())

    return df

df = load_data()

st.title("Failure Case Analysis (Customer Segmentation)")

# -------------------------
# 모델 학습
# -------------------------
@st.cache_resource
def train_model(df):
    X = df.drop(columns=["ID", "Segmentation"])
    y = df["Segmentation"]

    # 범주형 인코딩
    cat_cols = X.select_dtypes(include="object").columns
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    test_df = X_test.copy()
    test_df["Actual"] = y_test.values
    test_df["Predicted"] = y_pred

    # 그리드 서치를 안 쓴 이유 : 성능 높은 모델을 만드는 것이 아닌 실패 분석이 이 프로젝트의 목적이기 때문.
    # 위와 같은 이유로 딥러닝도 쓰지 않음

    return test_df, model

test_df, model = train_model(df)

# -------------------------
# 실패 사례 추출
# -------------------------
failure_df = test_df[test_df["Actual"] != test_df["Predicted"]]

st.subheader("실패 사례 요약")
st.write(f"총 실패 사례 수: **{len(failure_df)}**")

# -------------------------
# 실패 사례 텍스트 생성
# -------------------------
def generate_failure_text(row):
    return (
        f"고객은 {int(row['Age'])}세이며 소비 성향은 {row['Spending_Score']}이다. "
        f"직업은 {row['Profession']}이고 가족 수는 {int(row['Family_Size'])}명이다. "
        f"경력은 {int(row['Work_Experience'])}년이다. "
        f"실제 세그먼트는 {row['Actual']}였으나 "
        f"모델은 {row['Predicted']}로 잘못 예측했다."
    )

failure_texts = [generate_failure_text(row) for _, row in failure_df.iterrows()]

# 벡터 DB (FAISS)
@st.cache_resource
def build_vector_db(texts):
    embeddings = []
    for t in texts:
        emb = client.embeddings.create(
            model="text-embedding-3-small",
            input=t
        ).data[0].embedding
        embeddings.append(emb)

    embeddings = np.array(embeddings).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return index, embeddings

index, embeddings = build_vector_db(failure_texts)

# 실패 사례
st.subheader("실패 사례 챗봇")

question = st.text_input(
    "실패 사례 기반 질문을 입력하세요",
    placeholder="예: 고령 고객에서 오분류가 자주 발생하나요?"
)

def retrieve_context(query, k=5):
    q_emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    ).data[0].embedding

    q_emb = np.array([q_emb]).astype("float32")
    _, idxs = index.search(q_emb, k)

    return "\n".join([failure_texts[i] for i in idxs[0]])

if question:
    context = retrieve_context(question)

    SYSTEM_PROMPT = """
당신은 실패 사례 분석 전용 데이터 분석가입니다.
아래 실패 사례 요약에 포함된 정보만을 근거로 답변해주세요.

규칙:
1. 추측하지 말아주세요.
2. 일반론을 말하지 말아주세요.
3. 근거가 없으면 '분석 불가'라고 답해주세요.
"""

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
        max_tokens=400
    )

    st.markdown("### Answer")
    st.write(response.choices[0].message.content)
