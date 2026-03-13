import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))
from openai import OpenAI
from rag_engine import build_vectorstore, get_embeddings
from langchain_community.vectorstores import Chroma

CHROMA_DIR = os.path.join(os.path.dirname(__file__), "../data/chroma_db")

st.set_page_config(page_title="智慧城市问答", page_icon="🏙️", layout="wide")
st.title("🏙️ 智慧城市 RAG 问答系统")
st.caption("杭州交通 & 天气数据 · DeepSeek + ChromaDB")

@st.cache_resource
def load_vs():
    return build_vectorstore()

with st.spinner("加载向量库..."):
    vs = load_vs()
st.success("✅ 数据就绪：杭州1383个交通站点 + 实时天气")

with st.sidebar:
    st.header("💡 示例问题")
    examples = ["杭州今天天气怎么样？","本周会下雨吗？","杭州东站在哪里？","杭州有哪些地铁站？","明天适合出行吗？"]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state["preset"] = ex
    st.divider()
    st.markdown("**系统信息**")
    st.markdown("- 数据：OpenStreetMap + Open-Meteo")
    st.markdown("- 向量库：ChromaDB")
    st.markdown("- 模型：DeepSeek Chat")
    st.markdown("- Embedding：BGE-small-zh")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📄 查看参考数据"):
                for i, s in enumerate(msg["sources"]):
                    st.caption(f"[{i+1}] {s[:150]}...")

query = st.chat_input("输入问题，例如：杭州今天天气怎么样？")
if "preset" in st.session_state:
    query = st.session_state.pop("preset")

if query:
    st.session_state.messages.append({"role":"user","content":query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        docs = vs.similarity_search(query, k=4)
        sources = [d.page_content for d in docs]
        context = "\n\n".join([f"[文档{i+1}]\n{d.page_content}" for i,d in enumerate(docs)])

        client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
        placeholder = st.empty()
        full_text = ""
        for chunk in client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role":"system","content":"你是智慧城市数据助手，基于杭州城市数据用中文回答，数据外的如实说明。"},
                {"role":"user","content":f"城市数据：\n{context}\n\n问题：{query}"}
            ],
            max_tokens=512, stream=True,
        ):
            delta = chunk.choices[0].delta.content or ""
            full_text += delta
            placeholder.markdown(full_text + "▌")
        placeholder.markdown(full_text)

        with st.expander("📄 查看参考数据"):
            for i, s in enumerate(sources):
                st.caption(f"[{i+1}] {s[:150]}...")

    st.session_state.messages.append({"role":"assistant","content":full_text,"sources":sources})
