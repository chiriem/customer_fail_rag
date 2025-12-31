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
    df = df.drop("ID", axis=1)
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

    return test_df, model

test_df, model = train_model(df)

# -------------------------
# 실패 사례 추출
# -------------------------
failure_df = test_df[test_df["Actual"] != test_df["Predicted"]].copy()

st.subheader("실패 사례 요약")
st.write(f"총 실패 사례 수: **{len(failure_df)}**")
st.dataframe(failure_df, use_container_width=True)

# -------------------------
# 카테고리 매핑
# -------------------------
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

# -------------------------
# (리팩터링) 실패 문서: text + meta
# -------------------------
failure_docs = []
for _, row in failure_df.iterrows():
    failure_docs.append({
        "text": generate_failure_text(row),
        "Gender": str(row.get("Gender", "UNK")),
        "Profession": str(row.get("Profession", "UNK")),
        "Age": int(row.get("Age", 0)),
        "Spending_Score": str(row.get("Spending_Score", "UNK")),
        "Ever_Married": str(row.get("Ever_Married", "UNK")),
        "Graduated": str(row.get("Graduated", "UNK")),
    })

failure_texts = [d["text"] for d in failure_docs]

# -------------------------
# (리팩터링) 전체 실패 분포 요약(앵커)
# -------------------------
def summarize_failure_distribution(failure_df, topn=5):
    cols = ["Gender", "Profession", "Spending_Score", "Ever_Married", "Graduated"]
    parts = []
    for col in cols:
        if col not in failure_df.columns:
            continue
        vc = failure_df[col].value_counts(dropna=False)
        total = int(vc.sum()) if len(vc) else 0
        if total == 0:
            parts.append(f"- {col}: (값 없음)")
            continue

        top = vc.head(topn)
        txt = ", ".join([f"{k}:{int(v)}({(v/total)*100:.1f}%)" for k, v in top.items()])
        if len(vc) > topn:
            rest = int(vc.iloc[topn:].sum())
            txt += f", 기타:{rest}({(rest/total)*100:.1f}%)"
        parts.append(f"- {col}: {txt}")
    return "\n".join(parts)

# -------------------------
# (리팩터링) 벡터 DB (cosine similarity: normalize + IndexFlatIP)
# -------------------------
@st.cache_resource
def build_vector_db_with_meta(texts):
    if len(texts) == 0:
        # 실패 사례가 없을 경우 대비
        dummy = np.zeros((1, 1536), dtype="float32")
        faiss.normalize_L2(dummy)
        index = faiss.IndexFlatIP(dummy.shape[1])
        index.add(dummy)
        return index, dummy

    embs = []
    for t in texts:
        emb = client.embeddings.create(
            model="text-embedding-3-small",
            input=t
        ).data[0].embedding
        embs.append(emb)

    embs = np.array(embs, dtype="float32")
    faiss.normalize_L2(embs)  # cosine 준비
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    return index, embs

index, embeddings = build_vector_db_with_meta(failure_texts)

# 다양성 강제 선택: MMR + 쿼터
def mmr_diverse_select(
    q_emb: np.ndarray,
    cand_embs: np.ndarray,
    cand_idxs: np.ndarray,
    docs: list,
    k_final: int = 12,
    lambda_param: float = 0.75,
    max_per_gender: int = 7,
    max_per_profession: int = 3,
):
    """
    Greedy MMR + 쿼터로 다양성 확보
    - lambda_param 높을수록 query 유사도 우선
    - max_per_* 로 특정 성별/직업 쏠림 제한
    """
    q = q_emb.reshape(1, -1)
    sim_q = (cand_embs @ q.T).reshape(-1)  # cosine (IP)

    selected = []
    selected_embs = []
    gender_count = {}
    prof_count = {}

    order = np.argsort(-sim_q)
    cand_idxs = cand_idxs[order]
    cand_embs = cand_embs[order]
    sim_q = sim_q[order]

    for _ in range(min(k_final, len(cand_idxs))):
        best_i = None
        best_score = -1e9

        for i in range(len(cand_idxs)):
            idx = int(cand_idxs[i])
            if idx in selected:
                continue

            g = docs[idx].get("Gender", "UNK")
            p = docs[idx].get("Profession", "UNK")

            if gender_count.get(g, 0) >= max_per_gender:
                continue
            if prof_count.get(p, 0) >= max_per_profession:
                continue

            if len(selected_embs) == 0:
                div_pen = 0.0
            else:
                s = np.array(selected_embs, dtype="float32")  # (m, d)
                div_pen = float(np.max(s @ cand_embs[i].reshape(-1, 1)))  # max cosine

            score = lambda_param * float(sim_q[i]) - (1 - lambda_param) * div_pen
            if score > best_score:
                best_score = score
                best_i = i

        if best_i is None:
            break

        chosen_idx = int(cand_idxs[best_i])
        selected.append(chosen_idx)
        selected_embs.append(cand_embs[best_i])

        g = docs[chosen_idx].get("Gender", "UNK")
        p = docs[chosen_idx].get("Profession", "UNK")
        gender_count[g] = gender_count.get(g, 0) + 1
        prof_count[p] = prof_count.get(p, 0) + 1

    return selected

def retrieve_context_diverse(query, k_final=12, k_search=60):
    if len(failure_texts) == 0:
        return "[실패 사례가 없습니다]"

    q_emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    ).data[0].embedding
    q_emb = np.array(q_emb, dtype="float32").reshape(1, -1)
    faiss.normalize_L2(q_emb)

    # 1) 넉넉히 후보 검색
    k_search = min(k_search, len(failure_texts))
    scores, idxs = index.search(q_emb, k_search)
    cand_idxs = idxs[0]
    cand_embs = embeddings[cand_idxs]

    # 2) 다양성 강제 선택(MMR + 쿼터)
    #    - 성별은 최종의 절반 이상 한쪽이 차지하지 않게 제한(데이터가 한쪽뿐이면 자연히 그쪽으로 채워짐)
    #    - 직업은 최종의 1/4 이상 한 직업이 차지하지 않게 제한
    max_per_gender = max(1, k_final // 2)
    max_per_profession = max(1, k_final // 4)

    selected_idxs = mmr_diverse_select(
        q_emb=q_emb[0],
        cand_embs=cand_embs,
        cand_idxs=cand_idxs,
        docs=failure_docs,
        k_final=k_final,
        lambda_param=0.75,
        max_per_gender=max_per_gender,
        max_per_profession=max_per_profession,
    )

    dist = summarize_failure_distribution(failure_df)
    picked_text = "\n".join([failure_docs[i]["text"] for i in selected_idxs])

    context = f"""[전체 실패사례 분포 요약(전체 failure_df 기준)]
{dist}

[질문과 관련해 선택된 실패사례(다양성 보정 적용)]
{picked_text}
"""
    return context

# -------------------------
# 실패 사례 챗봇
# -------------------------
st.subheader("실패 사례 챗봇")

question = st.text_input(
    "실패 사례 기반 질문을 입력하세요. 실패 케이스에서 일부를 검색하여 대답하기 때문에 통계 관련 질문은 정확한 대답이 어렵습니다.",
    placeholder="예: 검색된 실패 사례에서 특정 직업군이 많은가요?"
)

if question:
    context = retrieve_context_diverse(question, k_final=12, k_search=60)

    SYSTEM_PROMPT = """
당신은 머신러닝 예측 실패 사례를 분석하는 데이터 분석가입니다.

아래에 제공되는 실패 사례 문서를 근거로 분석을 수행하세요.

분석 지침:
- 가능한 원인 가설을 제시하되, 반드시 문서의 내용과 연결하세요.
- 문서에 근거가 없는 내용은 "제공된 데이터만으로는 판단하기 어렵다"고 명시하세요.

추가 규칙(편향 완화):
- '전체 실패사례 분포 요약'은 전체 failure_df 기준이며, '선택된 실패사례'는 검색된 샘플입니다.
- 샘플에서 보이는 특징을 전체 특징으로 단정하지 말고, 전체 분포 요약과 비교해 표현하세요.
- 특정 성별/직업이 "전부"라고 말하려면, 전체 분포 요약에서 100%일 때만 그렇게 표현하세요.

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
        max_tokens=500
    )

    st.markdown("### Answer")
    st.write(response.choices[0].message.content)

# -------------------------
# (옵션) 디버깅/검증용: 실패 분포 요약 표시
# -------------------------
with st.expander("디버그: 실패 사례 분포 요약 보기"):
    st.text(summarize_failure_distribution(failure_df))
