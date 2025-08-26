import os
import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import DashScopeEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatTongyi
from langchain.agents import initialize_agent, Tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.docstore.document import Document

# 导入本地工具函数
from utils.incremental import (
    load_cached_html, save_cached_html, load_index,
    save_index, clear_storage, url_to_hash
)

# 初始化语言模型，启用流式输出并设置模型名称和温度
llm = ChatTongyi(streaming=True, model_name="qwen-max-0428", temperature=0)

# 获取用户输入的网页 URL，设置默认值
url = st.text_input("请输入网页 URL", "https://nginx.org/en/")
if not url:
    st.stop()

# 初始化文档切分器
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

# 初始化嵌入模型
embeddings = DashScopeEmbeddings(model="text-embedding-v2")

# 1. 尝试加载已有索引
vectorstore, indexed_urls = load_index(embeddings)

# 2. 处理新 URL
if url not in indexed_urls:
    try:
        # 缓存网页内容
        @st.cache_data(show_spinner=False)
        def load_or_cache_url(url: str):
            cached = load_cached_html(url)
            if cached:
                return cached
            loader = WebBaseLoader(url)
            html = loader.load()[0].page_content
            save_cached_html(url, html)
            return html

        raw = load_or_cache_url(url)
        doc = Document(page_content=raw, metadata={"source": url})
        chunks = splitter.split_documents([doc])
        if vectorstore is None:
            vectorstore = FAISS.from_documents(chunks, embeddings)
        else:
            vectorstore.add_documents(chunks)
        indexed_urls.append(url)
        save_index(vectorstore, indexed_urls)
        st.toast(f"已增量更新 {url}")
    except Exception as e:
        st.error(f"处理新 URL 时出错: {e}")
        st.stop()
else:
    st.toast("该 URL 已索引，跳过抓取")

# 构建基于对话的问答链
qa = ConversationalRetrievalChain.from_llm(llm, retriever=vectorstore.as_retriever())

# 初始化搜索工具
search = DuckDuckGoSearchRun()

def web_qa_func(question):
    try:
        input_data = {
            "question": question,
            "chat_history": st.session_state.history
        }
        return qa(input_data)["answer"]
    except Exception as e:
        return f"处理问题时出错: {e}"

tools = [
    Tool(
        name="search",
        func=search.run,
        description="当需要实时信息或网页内容时调用"
    ),
    Tool(
        name="web_qa",
        func=web_qa_func,
        description=f"基于已抓取的网页 {url} 回答用户问题"
    ),
]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True,
    handle_parsing_errors=True
)

# 实现多轮对话功能
if "history" not in st.session_state:
    st.session_state.history = []

q = st.chat_input("对网页内容提问")
if q:
    st.session_state.history.append((q, ""))
    try:
        response = agent.run(q)  # agent.run 接收字符串输入
        st.session_state.history[-1] = (q, response)
    except Exception as e:
        st.session_state.history[-1] = (q, f"处理问题时出错: {e}")

# 显示对话历史
for h in st.session_state.history:
    st.chat_message("user").write(h[0])
    st.chat_message("assistant").write(h[1])

# 清空缓存按钮
if st.sidebar.button("⚠️ 清空本地缓存"):
    clear_storage()
    st.rerun()