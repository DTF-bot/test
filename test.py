"""
RAG 文档问答系统 —— 上传PDF，基于文档内容智能问答
终端运行：streamlit run rag_app.py
"""
import streamlit as st
import requests
from pypdf import PdfReader
import re

# ====== 配置 ======
API_KEY = st.secrets.get("DEEPSEEK_API_KEY", "sk-ef66402792fb44eabec0114c46da732a")
API_URL = "https://api.deepseek.com/chat/completions"

# ====== 工具函数 ======

def extract_text_from_pdf(pdf_file):
    """从上传的 PDF 文件中提取所有文字"""
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def split_text(text, chunk_size=500):
    """把长文本按段落切成小块"""
    paragraphs = text.split("\n")
    chunks = []
    current = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current) + len(para) < chunk_size:
            current += para + "\n"
        else:
            if len(current.strip()) > 50:
                chunks.append(current.strip())
            current = para + "\n"
    if len(current.strip()) > 50:
        chunks.append(current.strip())
    return chunks


def search_chunks(chunks, query, top_k=3):
    """关键词匹配 + 打分，返回最相关的文本块"""
    keywords = re.findall(r'[一-鿿]+|[a-zA-Z]+', query.lower())

    scored = []
    for chunk in chunks:
        chunk_lower = chunk.lower()
        score = 0
        for kw in keywords:
            count = chunk_lower.count(kw)
            if count > 0:
                score += count * 10
            for i in range(len(kw) - 1):
                if kw[i:i+2] in chunk_lower:
                    score += 1
        scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    result = [c for s, c in scored[:top_k] if s > 0]
    if not result:
        result = chunks[:top_k]
    return result


def ask_llm(context_chunks, user_question):
    """把检索到的上下文 + 用户问题发给 DeepSeek"""
    context = "\n\n---\n\n".join(context_chunks)

    prompt = f"""你是一个文档助手。请严格基于以下文档内容回答用户问题。
如果文档中没有相关信息，请诚实地说"文档中未找到相关信息"。

文档内容：
{context}

用户问题：{user_question}

请用中文回答，简洁清晰。"""

    messages = [{"role": "user", "content": prompt}]

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "deepseek-chat",
        "messages": messages,
        "stream": False
    }
    resp = requests.post(API_URL, headers=headers, json=data)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


# ====== 界面 ======
st.set_page_config(page_title="📚 RAG 文档问答", page_icon="📚")
st.title("📚 RAG 智能文档问答系统")
st.caption("上传 PDF，基于文档内容精准问答 | 关键词匹配 + DeepSeek")

# 侧边栏
with st.sidebar:
    st.header("📄 上传文档")
    uploaded_file = st.file_uploader("选择 PDF 文件", type="pdf")

    if uploaded_file:
        st.success(f"已上传：{uploaded_file.name}")

        if st.button("🔨 建立知识库"):
            with st.spinner("正在解析文档..."):
                text = extract_text_from_pdf(uploaded_file)
                chunks = split_text(text)
                st.session_state["chunks"] = chunks
                st.session_state["index_ready"] = True
            st.success(f"✅ 知识库已就绪！共 {len(chunks)} 个段落")
            st.info(f"文档总字数：{len(text)}")

# 主区域
if st.session_state.get("index_ready"):
    st.success(f"📖 已加载 {len(st.session_state['chunks'])} 个段落，可以开始提问")

    question = st.text_input("🔍 基于文档提问...", placeholder="例如：这份文档主要讲了什么？")

    if question:
        with st.spinner("检索中..."):
            retrieved = search_chunks(st.session_state["chunks"], question)

            st.divider()
            st.caption(f"📎 匹配到 {len(retrieved)} 个相关段落：")
            for i, chunk in enumerate(retrieved):
                with st.expander(f"段落 {i+1}"):
                    st.text(chunk[:300] + ("..." if len(chunk) > 300 else ""))

        with st.spinner("AI 生成回答中..."):
            answer = ask_llm(retrieved, question)
            st.divider()
            st.markdown("### 🤖 AI 回答")
            st.write(answer)

else:
    st.info("👈 先在左侧上传 PDF 并点击「建立知识库」")
