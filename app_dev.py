import streamlit as st

import joblib
from llama_index.schema import QueryBundle

from llama_index import set_global_service_context

from plots import plot_embeddings

from utils import st_initilize_session_state_as_none

from model_utils import get_service_context, make_index, make_reducer

from query_utils import parse_query

st.set_page_config(
    page_title="AccRAG",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Session State
st_initilize_session_state_as_none(["index", 'reducer', 'embeddings_reduced'])

if "document_projections_done" not in st.session_state.keys():
    st.session_state["document_projections_done"] = False

context = get_service_context()
set_global_service_context(context)

make_index()
make_reducer()

############### Start App ################
st.header("Job interview Accenture ML Architect 👨‍💻🤖", divider='grey')
st.markdown("#### Visualise different RAG methods")

col3, col4 = st.columns([0.8, 0.2])
prompt = col3.text_input("Ask Anything About Accenture")

top_k    = col4.number_input("Number of Chunks", value=5, min_value=1, max_value=10, step=1)
strategy = col4.selectbox("Select RAG method",
                            ["Naive",
                            "Query Expansion - Multiple Qns",
                            "Query Expansion - Hypothetical Ans"])

response, box_height = parse_query(prompt, top_k)
with col3:

    st.text_area("Model Answer", value=response.response if response else None , height=box_height, disabled=True)

col5, _ ,col6 = st.columns([0.75, 0.05, 0.2])

# Create three columns
col7, col8, col9 = st.columns([2, 4, 2])

if response:

    with col7:
        st.write("### Node Texts")
        for node in response.source_nodes:
            st.text_area(node.id_, node.text, height=100)

# Your main content goes in the middle column
with col8, st.container(height=800):
    st.write("Embedding Space")
    plot_embeddings(prompt, response, context)