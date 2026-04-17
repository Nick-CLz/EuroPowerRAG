"""EuroPowerRAG — Streamlit Q&A interface.

Run:  streamlit run app.py
"""

from datetime import datetime, timedelta

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="EuroPowerRAG",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚡ EuroPowerRAG")
    st.caption("European Power Market Intelligence")
    st.divider()

    st.subheader("Filters")

    country_options = {
        "All countries": None,
        "Germany (DE_LU)": ["DE_LU"],
        "France (FR)": ["FR"],
        "Netherlands (NL)": ["NL"],
        "Great Britain (GB)": ["GB"],
        "Spain (ES)": ["ES"],
        "Belgium (BE)": ["BE"],
    }
    selected_country_label = st.selectbox("Country", list(country_options.keys()))
    country_filter = country_options[selected_country_label]

    doc_type_options = {
        "All types": None,
        "Price data": ["price_data"],
        "Generation data": ["generation_data"],
        "News": ["news"],
        "Reports": ["report"],
        "Prices + News": ["price_data", "news"],
    }
    selected_type_label = st.selectbox("Document type", list(doc_type_options.keys()))
    doc_type_filter = doc_type_options[selected_type_label]

    use_date_filter = st.checkbox("Filter by date range")
    date_from = date_to = None
    if use_date_filter:
        col1, col2 = st.columns(2)
        with col1:
            date_from = st.date_input("From", value=datetime.now() - timedelta(days=30))
        with col2:
            date_to = st.date_input("To", value=datetime.now())
        date_from = date_from.strftime("%Y-%m-%d") if date_from else None
        date_to = date_to.strftime("%Y-%m-%d") if date_to else None

    n_results = st.slider("Documents to retrieve", min_value=3, max_value=10, value=5)

    st.divider()
    if st.button("Run ingestion now", use_container_width=True):
        with st.spinner("Ingesting data..."):
            try:
                from ingest import main
                rc = main()
                if rc == 0:
                    st.success("Ingestion complete")
                else:
                    st.error("Ingestion finished with errors — check terminal")
            except Exception as e:
                st.error(f"Ingestion failed: {e}")

    st.divider()
    st.caption("Data: ENTSO-E · RSS feeds · PDF reports")
    st.caption("Model: claude-sonnet-4-6")

# ── Main ──────────────────────────────────────────────────────────────────────

st.title("European Power Market Q&A")
st.caption(
    "Ask questions about EU electricity prices, generation mix, and market news. "
    "Answers are grounded in ingested data with source citations."
)

# Example questions
with st.expander("Example questions"):
    examples = [
        "What was the average day-ahead price for Germany last week?",
        "What share of France's power comes from nuclear?",
        "How does wind contribute to European generation?",
        "What factors are driving electricity prices higher?",
        "Compare day-ahead prices between Spain and the Netherlands.",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{ex[:20]}"):
            st.session_state["prefill_query"] = ex

# Query input
prefill = st.session_state.pop("prefill_query", "")
query = st.text_input(
    "Your question",
    value=prefill,
    placeholder="e.g. What drove German baseload prices last week?",
)

# Run query
if query:
    with st.spinner("Retrieving context and generating answer..."):
        try:
            from src.pipeline import rag_chain

            result = rag_chain.query(
                question=query,
                country_filter=country_filter,
                doc_type_filter=doc_type_filter,
                date_from=date_from,
                date_to=date_to,
                n_results=n_results,
            )
        except Exception as e:
            st.error(f"Query failed: {e}")
            st.stop()

    # Answer
    st.divider()
    st.subheader("Answer")
    st.markdown(result.answer)

    # Source citations
    if result.sources:
        st.divider()
        st.subheader(f"Sources ({result.n_docs_retrieved} retrieved)")

        for i, doc in enumerate(result.sources, 1):
            m = doc.metadata
            label = (
                f"**[{i}]** {m.get('source', 'Unknown')} — "
                f"{m.get('doc_type', '?')} — "
                f"{m.get('country_name', m.get('country', '?'))} — "
                f"{m.get('date', 'date unknown')}"
            )
            with st.expander(label):
                st.text(doc.page_content[:600])
                if m.get("url"):
                    st.caption(f"URL: {m['url']}")
    else:
        st.info("No source documents retrieved. Run ingestion to populate the index.")

    # Query history
    if "query_history" not in st.session_state:
        st.session_state["query_history"] = []
    st.session_state["query_history"].insert(0, query)
    st.session_state["query_history"] = st.session_state["query_history"][:10]

# History in sidebar
if st.session_state.get("query_history"):
    with st.sidebar:
        st.divider()
        st.subheader("Recent queries")
        for past_q in st.session_state["query_history"][:5]:
            if st.button(past_q[:50] + ("..." if len(past_q) > 50 else ""), key=f"h_{past_q[:30]}"):
                st.session_state["prefill_query"] = past_q
                st.rerun()
