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
    df = df.drop("ID", axis = 1)
    df = df.dropna()

    return df

df = load_data()

st.title("Failure Case Analysis (Customer Segmentation)")

# -------------------------
# 모델 학습
# -------------------------
@st.cache_resource
def train_model(df):
    # RAG용 원본 데이터 (인코딩 전)
    X_raw = df.drop("Segmentation", axis=1)
    y = df["Segmentation"]

    # ML용 데이터 (인코딩 후)
    X_encoded = X_raw.copy()

    # 범주형 인코딩
    cat_cols = X_encoded.select_dtypes(include="object").columns
    for col in cat_cols:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))

    # 데이터 분할 (random_state를 고정하여 인덱스 동기화)
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.3, random_state=42
    )
    
    # RAG용 (원본 데이터 분할)
    _, X_test_raw, _, _ = train_test_split(
        X_raw, y, test_size=0.3, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # 결과 데이터프레임은 원본 데이터(X_test_raw)를 사용
    test_df = X_test_raw.copy()
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
st.dataframe(failure_df)

CATEGORY_MAP = {
    "Gender": {"Male": "남성", "Female": "여성"},
    "Ever_Married": {"Yes": "기혼", "No": "미혼"},
    "Graduated": {"Yes": "대학 졸업", "No": "대학 미졸업"},
    "Spending_Score": {
        "Low": "소비 성향이 낮음",
        "Average": "소비 성향이 보통",
        "High": "소비 성향이 높음"
    }
}

# 실패 사례 텍스트 생성
def generate_failure_text(row):
    gender = CATEGORY_MAP["Gender"].get(row["Gender"], row["Gender"])
    married = CATEGORY_MAP["Ever_Married"].get(row["Ever_Married"], row["Ever_Married"])
    graduated = CATEGORY_MAP["Graduated"].get(row["Graduated"], row["Graduated"])
    spending = CATEGORY_MAP["Spending_Score"].get(row["Spending_Score"], row["Spending_Score"])

    return (
        f"고객은 {int(row['Age'])}세, {gender}이며 {married} 상태이다. "
        f"학력은 {graduated}이고 {spending}이다. "
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
    placeholder="예: 실패 사례 중 제일 높은 나이를 알려주세요."
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
당신은 머신러닝 예측 실패 사례를 분석하는 데이터 분석가입니다.

아래에 제공되는 실패 사례 문서를 근거로 분석을 수행하세요.

분석 지침:
- 실패 사례에 공통적으로 나타나는 특징을 관찰하세요.
- 반복되는 패턴이 있다면 요약하세요.
- 가능한 원인 가설을 제시하되, 반드시 문서의 내용과 연결하세요.
- 문서에 근거가 없는 내용은 "제공된 데이터만으로는 판단하기 어렵다"고 명시하세요.

답변 형식:
관찰된 특징
- ...

반복되는 실패 패턴
- ...

가능한 원인 가설
- ...

답변은 분석가의 보고서처럼 작성하세요.
"""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"""
[실패 사례 요약]
{context}

위 문서를 근거로 다음 질문에 답변하세요.

[질문]
{question}
"""
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.2,
        max_tokens=400
    )

    st.markdown("### Answer")
    st.write(response.choices[0].message.content)
