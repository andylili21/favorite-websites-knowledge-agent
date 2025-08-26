import os, hashlib, json, shutil
from pathlib import Path
from typing import List
from langchain.schema import Document

CACHE_DIR = Path("storage/html_cache")
INDEX_DIR = Path("storage/faiss_index")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

def url_to_hash(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()

def load_cached_html(url: str) -> str | None:
    f = CACHE_DIR / f"{url_to_hash(url)}.html"
    return f.read_text() if f.exists() else None

def save_cached_html(url: str, html: str):
    (CACHE_DIR / f"{url_to_hash(url)}.html").write_text(html, encoding="utf-8")

def save_index(vectorstore, metas: List[str]):
    """把索引和 meta 一起落盘"""
    vectorstore.save_local(str(INDEX_DIR))
    (INDEX_DIR / "urls.json").write_text(json.dumps(metas, ensure_ascii=False))

def load_index(embeddings):
    """加载索引，不存在返回 None"""
    if not (INDEX_DIR / "index.faiss").exists():
        return None, []
    vs = FAISS.load_local(str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True)
    urls = json.loads((INDEX_DIR / "urls.json").read_text())
    return vs, urls

def clear_storage():
    """一键清空"""
    shutil.rmtree(CACHE_DIR, ignore_errors=True)
    shutil.rmtree(INDEX_DIR, ignore_errors=True)