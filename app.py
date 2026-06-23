import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.ensemble import IsolationForest
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

st.set_page_config(
    page_title="California Grid Monitor",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .stApp { background-color: #f4f5f0; }
    .block-container { padding: 1.5rem 2rem; }
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e4dd;
        border-radius: 10px;
        padding: 1.2rem 1.4rem;
    }
    .metric-label {
        font-size: 11px;
        color: #6b7a6b;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 600;
        color: #1a2e1a;
        line-height: 1.1;
    }
    .metric-unit {
        font-size: 13px;
        color: #6b7a6b;
        margin-left: 4px;
    }
    .anomaly-card {
        background: #ffffff;
        border: 1px solid #e2e4dd;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 10px;
    }
    .anomaly-time { font-size: 13px; color: #6b7a6b; }
    .anomaly-value { font-size: 18px; font-weight: 600; color: #1a2e1a; }
    .badge-up {
        background: rgba(220,53,53,0.1);
        color: #c0392b;
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 11px;
        font-weight: 500;
    }
    .badge-down {
        background: rgba(39,174,96,0.1);
        color: #1e6b3a;
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 11px;
        font-weight: 500;
    }
    .section-title {
        font-size: 11px;
        color: #6b7a6b;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 12px;
        margin-top: 24px;
    }
    .answer-box {
        background: #ffffff;
        border-left: 3px solid #1e6b3a;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        font-size: 13px;
        color: #1a2e1a;
        line-height: 1.7;
        margin-top: 8px;
    }
    .source-tag { font-size: 11px; color: #6b7a6b; margin-top: 6px; }
    h1, h2 { color: #1a2e1a !important; }
    .stButton button {
        background: #ffffff;
        color: #1a2e1a;
        border: 1px solid #e2e4dd;
        border-radius: 6px;
        font-size: 12px;
        padding: 4px 12px;
    }
    .stButton button:hover { background: #e8ede8; color: #1a2e1a; border-color: #1e6b3a; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv("/Users/ruchi/Desktop/eia_demand_raw.csv", parse_dates=["period"])
    df["hour"] = df["period"].dt.hour
    df["dayofweek"] = df["period"].dt.dayofweek
    df["rolling_mean"] = df["value"].rolling(window=24, min_periods=1).mean()
    df["rolling_std"] = df["value"].rolling(window=24, min_periods=1).std().fillna(0)
    df["deviation"] = df["value"] - df["rolling_mean"]
    features = df[["value", "hour", "dayofweek", "rolling_mean", "deviation"]].fillna(0)
    model = IsolationForest(contamination=0.05, random_state=42)
    df["anomaly"] = model.fit_predict(features)
    df["is_anomaly"] = df["anomaly"] == -1
    return df

@st.cache_resource
def load_rag():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(
        persist_directory="/Users/ruchi/Desktop/chroma_db",
        embedding_function=embeddings
    )
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt_template = """You are an energy grid analyst. Use the context below to explain the anomaly. Be specific and draw from the context provided.

Context:
{context}

Question: {question}

Answer:"""
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

df = load_data()
qa_chain = load_rag()
anomalies = df[df["is_anomaly"]]

st.markdown("## California grid monitor")
st.markdown(f"<span style='font-size:12px;color:#6b7a6b'>Live anomaly detection + AI explanation &nbsp;·&nbsp; {df['period'].min().strftime('%b %d')} – {df['period'].max().strftime('%b %d, %Y')}</span>", unsafe_allow_html=True)

st.markdown("<div class='section-title'>Overview</div>", unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""<div class='metric-card'>
        <div class='metric-label'>Avg demand</div>
        <div class='metric-value'>{int(df['value'].mean()):,}<span class='metric-unit'>MWh</span></div>
    </div>""", unsafe_allow_html=True)

with c2:
    st.markdown(f"""<div class='metric-card'>
        <div class='metric-label'>Peak demand</div>
        <div class='metric-value'>{int(df['value'].max()):,}<span class='metric-unit'>MWh</span></div>
    </div>""", unsafe_allow_html=True)

with c3:
    st.markdown(f"""<div class='metric-card'>
        <div class='metric-label'>Anomalies detected</div>
        <div class='metric-value'>{len(anomalies)}<span class='metric-unit'>hrs</span></div>
    </div>""", unsafe_allow_html=True)

with c4:
    st.markdown(f"""<div class='metric-card'>
        <div class='metric-label'>Anomaly rate</div>
        <div class='metric-value'>{len(anomalies)/len(df)*100:.1f}<span class='metric-unit'>%</span></div>
    </div>""", unsafe_allow_html=True)

st.markdown("<div class='section-title'>Hourly demand</div>", unsafe_allow_html=True)

fig, ax = plt.subplots(figsize=(14, 3.5))
fig.patch.set_facecolor("#ffffff")
ax.set_facecolor("#ffffff")
ax.plot(df["period"], df["value"], color="#1e6b3a", linewidth=1.5, alpha=0.9)
ax.fill_between(df["period"], df["value"], alpha=0.06, color="#1e6b3a")
ax.scatter(anomalies["period"], anomalies["value"],
           color="#c0392b", zorder=5, s=40, label="Anomaly")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax.tick_params(colors="#6b7a6b", labelsize=9)
for spine in ax.spines.values():
    spine.set_edgecolor("#e2e4dd")
ax.grid(axis="y", color="#e2e4dd", linewidth=0.5)
ax.set_ylabel("MWh", color="#6b7a6b", fontsize=10)
fig.tight_layout()
st.pyplot(fig)

st.markdown("<div class='section-title'>Detected anomalies</div>", unsafe_allow_html=True)

for _, row in anomalies.iterrows():
    direction = "above" if row["deviation"] > 0 else "below"
    badge_class = "badge-up" if row["deviation"] > 0 else "badge-down"
    sign = "+" if row["deviation"] > 0 else "-"

    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"""<div class='anomaly-card'>
            <div class='anomaly-time'>{row['period'].strftime('%a %b %d · %I:%M %p')}</div>
            <div class='anomaly-value'>{int(row['value']):,} MWh &nbsp;
                <span class='{badge_class}'>{sign}{abs(int(row['deviation'])):,} MWh vs avg</span>
            </div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)
        if st.button("Explain", key=str(row["period"])):
            with st.spinner("Analyzing..."):
                question = (
                    f"California electricity demand was {int(row['value']):,} MWh "
                    f"at {row['period'].strftime('%I:%M %p')} on "
                    f"{row['period'].strftime('%B %d')}, which is "
                    f"{abs(int(row['deviation'])):,} MWh {direction} the recent average. "
                    f"What grid or weather conditions could explain this?"
                )
                result = qa_chain.invoke({"query": question})
                sources = list(set([
                    doc.metadata.get("source", "").replace("https://", "").split("/")[0]
                    for doc in result["source_documents"]
                ]))
                st.markdown(f"""<div class='answer-box'>
                    {result['result']}
                    <div class='source-tag'>Sources: {" · ".join(sources)}</div>
                </div>""", unsafe_allow_html=True)