import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))
from openai import OpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
import glob

DATA_DIR   = os.path.join(os.path.dirname(__file__), "../data")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "../data/chroma_db")
os.makedirs(DATA_DIR, exist_ok=True)

st.set_page_config(page_title="智慧城市问答", page_icon="🏙️", layout="wide")
st.title("🏙️ 城市公共数据 RAG 智能问答平台")
st.caption("交通 & 天气数据 · DeepSeek + ChromaDB · 数据来源：OpenStreetMap + Open-Meteo")

@st.cache_resource(show_spinner="加载向量模型...")
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

@st.cache_resource(show_spinner="初始化知识库...")
def load_or_build_vs():
    embeddings = get_embeddings()
    # 已有向量库直接加载
    if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    # 没有则先爬数据再构建
    st.info("首次启动，正在爬取城市数据...")
    from scraper import main as scrape
    scrape()
    txt_files = glob.glob(os.path.join(DATA_DIR, "*_chunks_*.txt"))
    all_texts = []
    for path in txt_files:
        with open(path, "r", encoding="utf-8") as f:
            blocks = [b.strip() for b in f.read().split("---") if b.strip()]
            all_texts.extend(blocks)
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
    docs = splitter.create_documents(all_texts)
    return Chroma.from_documents(docs, embedding=embeddings, persist_directory=CHROMA_DIR)

vs = load_or_build_vs()
st.success("✅ 知识库就绪")

with st.sidebar:
    st.header("💡 示例问题")
    examples = [
        "杭州今天天气怎么样？",
        "本周会下雨吗？",
        "杭州东站在哪里？",
        "杭州有哪些地铁站？",
        "明天适合出行吗？",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state["preset"] = ex
    st.divider()
    st.markdown("**系统信息**")
    st.markdown("- 数据：OpenStreetMap + Open-Meteo")
    st.markdown("- 向量库：ChromaDB")
    st.markdown("- 模型：DeepSeek Chat")
    st.markdown("- Embedding：BGE-small-zh")
    st.divider()
    if st.button("🔄 刷新城市数据", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()

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
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        docs = vs.similarity_search(query, k=4)
        sources = [d.page_content for d in docs]
        context = "\n\n".join([f"[文档{i+1}]\n{d.page_content}" for i, d in enumerate(docs)])
        api_key = os.getenv("DEEPSEEK_API_KEY") or st.secrets.get("DEEPSEEK_API_KEY", "")
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        placeholder = st.empty()
        full_text = ""
        for chunk in client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是智慧城市数据助手，基于城市数据用中文回答，数据外的如实说明。"},
                {"role": "user", "content": f"城市数据：\n{context}\n\n问题：{query}"}
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

    st.session_state.messages.append({
        "role": "assistant", "content": full_text, "sources": sources
    })
