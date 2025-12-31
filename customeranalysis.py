import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# =========================
# Streamlit 설정
# =========================
st.set_page_config(page_title="Failure Analysis RAG", layout="wide")
st.title("📉 실패 사례 기반 분석 & Q&A")

# =========================
# 데이터 로드
# =========================
@st.cache_data
def load_data():
    return pd.read_csv("data/train.csv")

df = load_data()

# =========================
# 컬럼 정의
# =========================
NUMERIC_COLS = ["Age", "Work_Experience", "Family_Size"]
CATEGORICAL_COLS = [
    "Gender",
    "Ever_Married",
    "Graduated",
    "Profession",
    "Spending_Score"
]
TARGET = "Segmentation"

# =========================
# 인코딩 + 디코딩 맵 생성
# =========================
df_ml = df.copy()
decode_maps = {}

for col in CATEGORICAL_COLS + [TARGET]:
    le = LabelEncoder()
    df_ml[col] = le.fit_transform(df_ml[col].astype(str))
    decode_maps[col] = dict(zip(le.transform(le.classes_), le.classes_))

X = df_ml[NUMERIC_COLS + CATEGORICAL_COLS]
y = df_ml[TARGET]

# =========================
# Train / Validation 분할
# =========================
X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
    X, y, df.index,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# 모델 학습
# =========================
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# =========================
# 실패 사례 생성
# =========================
val_pred = model.predict(X_val)

df_val = df.loc[idx_val].copy()
df_val["Actual"] = y_val.values
df_val["Predicted"] = val_pred

df_failure = df_val[df_val["Actual"] != df_val["Predicted"]].copy()

st.subheader("❌ 실패 사례 수")
st.metric("Failures", len(df_failure))

# =========================
# 실패 텍스트 생성
# =========================
def generate_failure_text(row):
    return (
        f"{int(row['Age'])}세 고객으로 "
        f"성별은 {decode_maps['Gender'].get(row['Gender'], 'Unknown')}이며 "
        f"{decode_maps['Ever_Married'].get(row['Ever_Married'], 'Unknown')} 상태이다. "
        f"{decode_maps['Graduated'].get(row['Graduated'], 'Unknown')}이고 "
        f"직업은 {decode_maps['Profession'].get(row['Profession'], 'Unknown')}이다. "
        f"소비 성향은 {decode_maps['Spending_Score'].get(row['Spending_Score'], 'Unknown')}이며 "
        f"가족 수는 {int(row['Family_Size'])}명, "
        f"경력은 {int(row['Work_Experience'])}년이다. "
        f"실제 세그먼트는 "
        f"{decode_maps['Segmentation'].get(row['Actual'], 'Unknown')}이나 "
        f"모델은 "
        f"{decode_maps['Segmentation'].get(row['Predicted'], 'Unknown')}로 잘못 예측했다."
    )

failure_texts = df_failure.apply(generate_failure_text, axis=1).tolist()

# =========================
# Vector DB (FAISS)
# =========================
embedder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedder.encode(failure_texts)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

def retrieve_context(query, k=5):
    q_emb = embedder.encode([query])
    _, idxs = index.search(np.array(q_emb), k)
    return "\n\n".join([failure_texts[i] for i in idxs[0]])

# =========================
# OpenAI
# =========================
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# =========================
# Q&A UI
# =========================
st.subheader("💬 실패 사례 기반 Q&A")

question = st.text_input("실패 사례에 대해 질문하세요")

if question and len(failure_texts) > 0:
    context = retrieve_context(question)

    messages = [
        {
            "role": "system",
            "content": (
                "당신은 실패 사례 분석 전용 데이터 분석가입니다. "
                "아래 실패 사례 요약에 포함된 정보만을 근거로 답변하세요. "
                "근거가 없으면 '분석 불가'라고 답변하세요."
            )
        },
        {
            "role": "user",
            "content": f"[실패 사례]\n{context}\n\n[질문]\n{question}"
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0
    )

    st.markdown("### 📌 답변")
    st.write(response.choices[0].message.content)
