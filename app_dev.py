import streamlit as st

from ui import st_header

from utils import (
    st_initilize_session_state_as_none,
    prepare_projections_df
)

from constants import (
    PLOT_SIZE,
    ABOUT_THIS_APP,
    CHUNK_EXPLAINER,
    BUILD_VDB_LOADING_MSG,
    VISUALISE_LOADING_MSG
)

st.set_page_config(
    page_title="AccRAG",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Session State
st_initilize_session_state_as_none(["document", "chroma", "filtered_df", "document_projections"])

if "document_projections_done" not in st.session_state.keys():
    st.session_state["document_projections_done"] = False

st_header()

if st.session_state['document'] is None:
    col1, col2 = st.columns(2)
    col1.markdown("### 1. Upload your PDF 📄")
    col1.markdown("For this demo, a 10-20 page PDF is recommended.")
    uploaded_file = col1.file_uploader("Upload your PDF", label_visibility="collapsed", type='pdf')
    col1.markdown("### 2. Configurations (Optional) 🔧")
    st.session_state["chunk_size"] = col1.number_input("Chunk Size", value = 1000, step = 50)
    st.session_state["chunk_overlap"] = col1.number_input("Chunk Overlap", step = 50)
    st.session_state["embedding_model"] = col1.selectbox("Select your embedding model",
                                  ["all-MiniLM-L6-v2",
                                   "text-embedding-ada-002",
                                  # "gte-large"
                                   ])

    col1.markdown("### 3. Build VectorDB ⚡️")
    if col1.button("Build"):
        st.session_state['document'] = uploaded_file
        st.rerun()

    with col2.expander("**About this application**"):
        st.success(ABOUT_THIS_APP)

    with col2.expander("**EXPLAINER:** What is chunk size/overlap?"):
        st.info(CHUNK_EXPLAINER)