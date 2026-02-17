import requests
import streamlit as st

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="POS RAG", layout="wide")

st.title("Personal Operating System RAG")
st.caption("Modes: Study, Build, Career, Life, Health. Answers are grounded in your sources with citations.")

col1, col2 = st.columns([1, 1])

with col1:
    mode = st.selectbox("Mode", ["study", "build", "career", "life", "health"], index=0)
    question = st.text_area("Ask a question", height=120, placeholder="Example: What does the rubric require for Lab 5?")
    debug = st.checkbox("Debug (show scores)", value=False)

with col2:
    st.subheader("Indexing")
    st.write("Put files into: data/sources/<mode>/ then reindex.")
    reindex_mode = st.selectbox("Reindex mode", ["study", "build", "career", "life", "health", "all"], index=5)
    if st.button("Reindex now"):
        payload = {"modes": None} if reindex_mode == "all" else {"modes": [reindex_mode]}
        r = requests.post(f"{API_BASE}/reindex", json=payload, timeout=600)
        st.json(r.json())

    if st.button("Status"):
        r = requests.get(f"{API_BASE}/status", timeout=30)
        st.json(r.json())

st.divider()

if st.button("Ask"):
    if not question.strip():
        st.warning("Type a question first.")
    else:
        payload = {"mode": mode, "question": question, "strict": True, "debug": debug}
        r = requests.post(f"{API_BASE}/query", json=payload, timeout=120)
        data = r.json()

        if not data.get("ok"):
            st.error(data.get("error", "Unknown error"))
        else:
            refused = data.get("refused", False)
            if refused:
                st.error(data["answer"])
            else:
                st.success("Answer generated from your sources.")
                st.text(data["answer"])

            st.subheader("Citations")
            citations = data.get("citations", [])
            if not citations:
                st.write("No citations available yet. Add sources and reindex.")
            else:
                for i, c in enumerate(citations, start=1):
                    title = f"{i}) {c['source_path']}"
                    meta = []
                    if c.get("heading"):
                        meta.append(f"heading: {c['heading']}")
                    if c.get("page"):
                        meta.append(f"page: {c['page']}")
                    meta.append(f"score: {c['score']:.3f}")
                    with st.expander(title + " (" + ", ".join(meta) + ")"):
                        st.write(c["snippet"])

            if debug and data.get("debug"):
                st.subheader("Debug")
                st.json(data["debug"])