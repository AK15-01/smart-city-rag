import os, glob
from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))

DATA_DIR   = os.path.join(os.path.dirname(__file__), "../data")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "../data/chroma_db")

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

def get_embeddings():
    print("[向量] 加载embedding模型（首次下载~90MB，请稍候）...")
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

def build_vectorstore(force_rebuild=False):
    embeddings = get_embeddings()
    if os.path.exists(CHROMA_DIR) and not force_rebuild:
        print("[向量库] 加载已有向量库...")
        return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

    txt_files = glob.glob(os.path.join(DATA_DIR, "*_chunks_*.txt"))
    if not txt_files:
        raise FileNotFoundError("未找到数据！请先运行：python src/scraper.py")

    all_texts = []
    for path in txt_files:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        blocks = [b.strip() for b in content.split("---") if b.strip()]
        all_texts.extend(blocks)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=80,
        separators=["\n\n", "\n", "，", "。"],
    )
    docs = splitter.create_documents(all_texts)
    print(f"[向量库] 共 {len(docs)} 个文档块，向量化中...")

    vs = Chroma.from_documents(
        documents=docs, embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    print(f"[向量库] ✅ 构建完成 → {CHROMA_DIR}")
    return vs

def ask(query, vs):
    print(f"\n❓ 问题：{query}")
    docs = vs.similarity_search(query, k=4)
    print(f"[检索] 找到 {len(docs)} 个相关块")

    context = "\n\n".join([f"[文档{i+1}]\n{d.page_content}" for i, d in enumerate(docs)])
    system = "你是智慧城市数据助手，基于提供的杭州城市数据用中文回答问题，数据外的信息如实说明。"
    user   = f"城市数据：\n{context}\n\n问题：{query}"

    print("[AI助手] ", end="", flush=True)
    full = ""
    for chunk in client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
        max_tokens=512, stream=True,
    ):
        delta = chunk.choices[0].delta.content or ""
        print(delta, end="", flush=True)
        full += delta
    print("\n")
    return full

def main():
    print("="*50)
    print("智慧城市 RAG 问答系统")
    print("="*50)
    vs = build_vectorstore()

    for q in ["杭州今天天气怎么样？", "杭州有哪些火车站？", "本周会下雨吗？"]:
        ask(q, vs)
        print("-"*40)

    print("💬 交互模式（输入 q 退出）：")
    while True:
        query = input("\n问题> ").strip()
        if query.lower() in ("q","quit","退出"): break
        if query: ask(query, vs)

if __name__ == "__main__":
    main()
