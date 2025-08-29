# rag_k8s.py
"""
RAG application for Kubernetes documentation.
Provides a class-based interface for indexing and querying docs.
"""

import os
import re
import json
import pathlib
from typing import List, Dict

import numpy as np
from tqdm import tqdm

# Optional dependencies
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class KubernetesRAG:
    DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    DEFAULT_GEN_MODEL = "google/flan-t5-base"
    DEFAULT_MAX_CHARS = 4000

    def __init__(self, docs_dir: str, store_dir: str = "./k8s_rag_store"):
        self.docs_dir = docs_dir
        self.store_dir = store_dir
        self.emb_model = None
        self.gen_model = None
        self.index = None
        self.texts = []
        self.meta = []

        os.makedirs(store_dir, exist_ok=True)

    # ---------------- Document Utilities ----------------
    @staticmethod
    def clean_text(s: str) -> str:
        if not s:
            return ""
        s = re.sub(r"https?://\S+", "", s)
        s = s.replace("\u00a0", " ")
        s = re.sub(r"\s+", " ", s)
        return s.strip()

    @staticmethod
    def chunk_text(text: str, max_tokens: int = 300, overlap: int = 50) -> List[str]:
        words = text.split()
        if not words:
            return []
        chunks = []
        step = max(1, max_tokens - overlap)
        for i in range(0, len(words), step):
            chunk = words[i : i + max_tokens]
            if chunk:
                chunks.append(" ".join(chunk))
        return chunks

    @staticmethod
    def read_text(path: str) -> str:
        ext = pathlib.Path(path).suffix.lower()
        if ext in {".txt", ".md"}:
            return pathlib.Path(path).read_text(encoding="utf-8", errors="ignore")
        if ext in {".html", ".htm"}:
            if BeautifulSoup is None:
                raise RuntimeError("beautifulsoup4 is required to parse HTML.")
            html = pathlib.Path(path).read_text(encoding="utf-8", errors="ignore")
            soup = BeautifulSoup(html, "lxml")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            return soup.get_text(" ")
        if ext == ".pdf":
            if PdfReader is None:
                raise RuntimeError("pypdf is required to parse PDFs.")
            reader = PdfReader(path)
            pages = []
            for p in reader.pages:
                pages.append(p.extract_text() or "")
            return "\n".join(pages)
        raise ValueError(f"Unsupported file type: {ext}")

    def discover_files(self) -> List[str]:
        files = []
        for root, _, fnames in os.walk(self.docs_dir):
            for f in fnames:
                if pathlib.Path(f).suffix.lower() in {".txt", ".md", ".html", ".htm", ".pdf"}:
                    files.append(os.path.join(root, f))
        return sorted(files)

    # ---------------- Indexing ----------------
    def build_index(self, emb_model_name: str = None, max_tokens: int = 300, overlap: int = 50):
        emb_model_name = emb_model_name or self.DEFAULT_EMBED_MODEL
        print(f"Loading embedding model: {emb_model_name}")
        self.emb_model = SentenceTransformer(emb_model_name)

        files = self.discover_files()
        if not files:
            raise SystemExit(f"No supported files found under {self.docs_dir}")

        meta, texts = [], []

        print(f"Reading and chunking {len(files)} files...")
        for path in tqdm(files):
            try:
                raw = self.read_text(path)
            except Exception as e:
                print(f"[WARN] Skipping {path}: {e}")
                continue
            raw = self.clean_text(raw)
            chunks = self.chunk_text(raw, max_tokens=max_tokens, overlap=overlap)
            for j, ch in enumerate(chunks):
                if len(ch) < 20:
                    continue
                texts.append(ch)
                meta.append({"source_path": os.path.relpath(path, self.docs_dir), "chunk_id": j})

        if not texts:
            raise SystemExit("No text chunks produced. Check your docs content.")

        print(f"Encoding {len(texts)} chunks...")
        embs = self.emb_model.encode(texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
        dim = embs.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embs)

        # Save FAISS + metadata
        faiss.write_index(index, os.path.join(self.store_dir, "index.faiss"))
        with open(os.path.join(self.store_dir, "texts.jsonl"), "w", encoding="utf-8") as f:
            for t in texts:
                f.write(json.dumps({"text": t}) + "\n")
        with open(os.path.join(self.store_dir, "meta.jsonl"), "w", encoding="utf-8") as f:
            for m in meta:
                f.write(json.dumps(m) + "\n")

        print(f"Indexed {len(texts)} chunks.")

        # Save in-memory
        self.index = index
        self.texts = texts
        self.meta = meta

    # ---------------- Query ----------------
    def load_index(self):
        if self.index is not None:
            return
        self.index = faiss.read_index(os.path.join(self.store_dir, "index.faiss"))
        self.texts = [json.loads(l)["text"] for l in open(os.path.join(self.store_dir, "texts.jsonl"))]
        self.meta = [json.loads(l) for l in open(os.path.join(self.store_dir, "meta.jsonl"))]
        self.emb_model = SentenceTransformer(self.DEFAULT_EMBED_MODEL)

    def retrieve(self, question: str, top_k: int = 5) -> List[Dict]:
        self.load_index()
        q_emb = self.emb_model.encode([question], normalize_embeddings=True)
        D, I = self.index.search(np.array(q_emb).astype("float32"), top_k)
        hits = []
        for rank, (idx, score) in enumerate(zip(I[0], D[0])):
            if idx == -1:
                continue
            hits.append({"rank": rank + 1, "score": float(score), "text": self.texts[idx], "meta": self.meta[idx]})
        return hits

    @staticmethod
    def build_context(hits: List[Dict], max_chars: int = DEFAULT_MAX_CHARS) -> str:
        out, total = [], 0
        for h in hits:
            t = h["text"].strip()
            if not t:
                continue
            if total + len(t) + 100 > max_chars:
                break
            out.append(f"[Source: {h['meta']['source_path']}#chunk{h['meta']['chunk_id']}]\n{t}")
            total += len(t) + 100
        return "\n\n".join(out)
    def load_generation_model(self, gen_model_name: str = None):
        if self.gen_model is None or self.gen_tokenizer is None:
            gen_model_name = gen_model_name or self.DEFAULT_GEN_MODEL
            print(f"Loading generation model: {gen_model_name}")
            self.gen_tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
            self.gen_model = AutoModelForSeq2SeqLM.from_pretrained(gen_model_name)
    def generate_answer(self, question: str, top_k: int = 5, gen_model_name: str = None, max_new_tokens: int = 256) -> str:
        self.load_generation_model(gen_model_name)
        hits = self.retrieve(question, top_k)
        context = self.build_context(hits)
        # print("Context for generation:\n", context)
        prompt = (
            f"You are a helpful assistant for Kubernetes documentation. "
            f"Answer only using the provided context. If the answer isn't present, say 'I couldn't find that in the docs.'\n\n"
            f"Question: {question}\n\nContext:\n{context}\n\nAnswer:"
        )
        inputs = self.gen_tokenizer(prompt, return_tensors="pt", truncation=True)
        out = self.gen_model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.gen_tokenizer.decode(out[0], skip_special_tokens=True)

# from rag_k8s import KubernetesRAG

# Initialize with local docs folder
# rag = KubernetesRAG(docs_dir="./k8s_docs", store_dir="./k8s_rag_store")

# # Build index (run once)
# # rag.build_index()


# print("--------------------------------------------------------------------")
# # Query
# while(1):
#     question = input("Enter your Kubernetes question (or 'exit' to quit): ")
#     if question.lower() in {'exit', 'quit'}:
#         break
#     answer = rag.generate_answer(question)
#     print("-----------------------------------------------------------------------------------")
#     print(answer)
# # answer = rag.generate_answer("whst is a pod in kubernetes?")


# # print("-----------------------------------------------------------------------------------")
# # print(answer)
