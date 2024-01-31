import streamlit as st

from model_utils import (
    make_service_context,
    make_index,
    make_embeddings,
    load_models,
    describe_cluster,
    parse_query,
    plot_embeddings,
    st_initilize_session_state_as_none,
)

st.set_page_config(
    page_title="RAG",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

static_objects = [
    "index",
    'reducer',
    'embeddings_reduced',
    'context',
    'llm',
    'embed_model',
    ]

st_initilize_session_state_as_none(static_objects)
load_models()

# Start Rendering
st.markdown("<h1 style='text-align: center; color: white;'>Visual RAG Inspector with Llama Index 🦙</h1>", unsafe_allow_html=True)
context_str = f'Curently configured with LLM: {st.session_state["llm"].model_name} and Embedding Model: {st.session_state["embed_model"].model_name}'
st.markdown(f"<h5 style='text-align: center; color: white;'>{context_str}</h5>", unsafe_allow_html=True)

# App Settings
st.markdown("<br><br>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1,1,2])
with col1:
    st.markdown("#### Upload PDF File")
    uploaded_pdf = st.file_uploader("Upload a PDF", type="pdf", accept_multiple_files=False)
with col2:
    st.markdown("#### RAG Stettings")
    chunk_size = st.number_input("Chunk Size", value=1024, min_value=128, max_value=2048, step=1)
    top_k = st.number_input("Number of Chunks for RAG", value=3, min_value=1, max_value=10, step=1)
    cluster_chunks = st.number_input("Number of Chunks to Describe Cluster", value=3, min_value=1, max_value=10, step=1)
with col3:
    st.markdown("#### LLM Interaction")
    prompt = st.text_input(label = "Question Regarding PDF")

    response, box_height = parse_query(prompt, top_k)

    st.text_area(
        label    = "Model Answer",
        value    = response.response if response is not None else None,
        height   = box_height if box_height is not None else 120,
        disabled = True,
        )

with col1:


        # Create a button
    if st.button('Click Here to Build the App!', use_container_width=True):
        make_service_context(chunk_size)
        make_index(uploaded_pdf)
        make_embeddings()

# Visual Inspection
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("#### Visual Inspection")

col5, col6, col7 = st.columns([2, 4, 2])

with col6, st.container(height=600):
    st.write("Embedding Space")

    if st.session_state["embeddings_reduced"] is not None:
        plot_embeddings(prompt, response)

with col5, st.container(height=600):
    st.write("")
    st.write("Identified Clusters")
    if st.session_state["embeddings_reduced"] is not None:
        for cluster in st.session_state["embeddings_reduced"].cluster.unique():

            #None assigned nodes
            if cluster == '-1': continue

            desc = describe_cluster(cluster, cluster_chunks)
            st.text_area(f'Cluster: {cluster}', desc, height=30)

with col7, st.container(height=600):
    st.write("")
    st.write("Retrieved Nodes")
    if response is not None:
        for node in response.source_nodes:
                st.text_area(f'None ID: {node.id_}, Score: {node.score:.5f}', node.text, height=100)