# 🏙️ 城市公共数据 RAG 智能问答平台
# 在线演示： https://smart-city-rag-vbdpgezze9ngchrxj92uxs.streamlit.app
一套可快速接入任意城市公共数据的 AI 智能问答平台，基于 RAG 架构，将城市交通、天气等开放数据转化为可对话的知识库。支持一键切换城市，适用于运营商、政务平台、智慧城市集成商等场景的 AI 能力落地。

# 技术栈
- LLM：DeepSeek Chat
- 向量数据库：ChromaDB
- Embedding：BAAI/bge-small-zh-v1.5
- RAG框架：LangChain
- 前端：Streamlit
- 数据源：OpenStreetMap + Open-Meteo（免费开放）
- 部署环境：Ubuntu + Python 3.12

# 快速开始
```bash
python3 -m venv venv && source venv/bin/activate
pip install langchain langchain-community langchain-huggingface langchain-chroma langchain-text-splitters chromadb openai streamlit python-dotenv requests sentence-transformers
echo "DEEPSEEK_API_KEY=你的key" > .env
python src/scraper.py
streamlit run src/app.py
```

# 适用场景
- 运营商政企业务：智慧城市 AI 问答模块
- 数字政务：市民服务智能助手  
- 城市数据平台：自然语言查询接口
