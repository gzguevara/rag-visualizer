import streamlit as st

import pandas as pd
import numpy as np

import umap

from llama_index import (
    ServiceContext,
    download_loader,
    VectorStoreIndex,
    PromptTemplate,
    PromptHelper,
)

from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.prompts.prompts import PromptTemplate
from llama_index.text_splitter import SentenceSplitter
from llama_index.llms import HuggingFaceLLM

from constants import (
    MODEL,
    HF_TOKEN,
    EMB_MODEL,
    PDF_DIR ,
    SYSTEM_PROMPT,
)

from sklearn.cluster import DBSCAN

import plotly.express as px

def st_initilize_session_state_as_none(key_list):
    for key in key_list:
        if key not in st.session_state:
            st.session_state[key] = None

@st.cache_resource
def load_models():

    query_wrapper_prompt = PromptTemplate("{query_str} [/INST]")

    generate_kwargs = {
        "temperature": 0.7,
        "do_sample": True,
        }

    model_kwargs={
        'load_in_8bit':True,
        'token':HF_TOKEN,
        }

    tokenizer_kwargs={
        "max_length": 4096,
        }

    # Create and dl embeddings instance
    embed_model=HuggingFaceEmbedding(
        model_name=EMB_MODEL,
        )

    # Create a HF LLM using the llama index wrapper
    llm = HuggingFaceLLM(

        system_prompt=SYSTEM_PROMPT,
        query_wrapper_prompt=query_wrapper_prompt,
        is_chat_model=False,

        tokenizer_name=MODEL,
        model_name=MODEL,

        generate_kwargs=generate_kwargs,
        tokenizer_kwargs=tokenizer_kwargs,
        model_kwargs=model_kwargs,
    )

    st.session_state["llm"] = llm
    st.session_state["embed_model"] = embed_model

def make_service_context(chunk_size):

    text_splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=50,
        include_metadata=True,
    )

    prompt_helper = PromptHelper(
        context_window=4096,
        num_output=256,
        chunk_overlap_ratio=0.1,
        chunk_size_limit=None,
    )

    service_context = ServiceContext.from_defaults(
        llm=st.session_state["llm"],
        prompt_helper=prompt_helper,
        embed_model=st.session_state["embed_model"],
        text_splitter=text_splitter,
    )

    st.session_state["context"] = service_context

def make_index(pdf_buffer):

    if pdf_buffer is not None:

        with open(PDF_DIR, "wb") as f:
            f.write(pdf_buffer.getbuffer())

        loader = download_loader("PDFReader")
        loader = loader()

        documents = loader.load_data(file=PDF_DIR)

        index = VectorStoreIndex.from_documents(
            documents=documents,
            service_context=st.session_state['context'],
        )

        st.session_state["index"] = index

def make_embeddings():

    idx=[]
    embs=[]

    index = st.session_state["index"]

    for key, emb in index._vector_store._data.embedding_dict.items():

        idx.append(key)
        embs.append(emb)

    reducer = umap.UMAP(n_components=3, random_state=42)
    emb_red = reducer.fit_transform(embs)

    dbscan = DBSCAN(eps=0.5, min_samples=5)
    clus   = dbscan.fit_predict(emb_red)

    embs = pd.DataFrame(embs, index=idx, columns=[f'emb{i}' for i in range(len(emb))])
    embs[['col0', 'col1', 'col2']] = emb_red
    embs['cluster'] = clus

    st.session_state["reducer"] = reducer
    st.session_state["embeddings_reduced"] = embs

@st.cache_data
def describe_cluster(cluster, cluster_chunks):

    data = st.session_state["embeddings_reduced"]
    index = st.session_state["index"]
    llm = st.session_state["llm"]

    chunks = data.loc[data.cluster==cluster].sample(cluster_chunks, random_state=42).index

    cluster_sample = ''

    for chunk in chunks:

        cluster_sample += index._docstore.to_dict()['docstore/data'][chunk]['__data__']['text']

    sum_prompt = '\n above you see a few paragraphs of a text cluster. Summarize in one sentence the cluster starting with the phrase "The cluster relates to"'

    summary = llm.complete(cluster_sample + sum_prompt)

    return summary.text

@st.cache_resource
def get_query_engine(num_chuncks):

    index = st.session_state["index"]
    engine = index.as_query_engine(similarity_top_k=num_chuncks)

    return engine

@st.cache_data
def parse_query(prompt, num_chuncks):

    if prompt != '':

        engine = get_query_engine(num_chuncks)

        response = engine.query(prompt)

        box_height = min(300, 20 + len(response.response) // 5) if response.response else 100

        return response, box_height

    return None, None

def plot_embeddings(prompt, response):

    data = st.session_state["embeddings_reduced"]
    data.cluster = data.cluster.astype(str)

    if prompt !=  None and response !=  None:

        reducer = st.session_state["reducer"]
        context = st.session_state["context"]

        prompt_emb = context.embed_model.get_agg_embedding_from_queries([prompt])
        prompt_emb = reducer.transform([prompt_emb])
        prompt_emb = pd.DataFrame(prompt_emb, index=['prompt'], columns=['col0', 'col1', 'col2'])

        data = pd.concat([data, prompt_emb])

        nodes = [node.id_ for node in response.source_nodes]
        data['class'] = np.where(data.index.isin(nodes), 'retrieved', 'node')
        data.loc[data.index=='prompt', 'class'] = 'prompt'
        data.loc[data.index=='prompt', 'cluster'] = 'prompt'

    fig = px.scatter_3d(
        data_frame=data,
        x='col0',
        y='col1',
        z='col2',
        color='cluster',
        hover_name=data.index,
        symbol='class' if 'class' in data.columns else None
        )

    fig.update_traces(marker_size=3)
    fig.update_layout(width=850, height=500)
    fig.update_traces(showlegend=False)
    # Make non assigned chunks grey
    fig.for_each_trace(lambda t: t.update(marker=dict(color='grey')) if '-1' in t['legendgroup'] else t)
    fig.for_each_trace(lambda t: t.update(marker=dict(color='#75FFB5')) if t['marker']['color'] == '#000004' else t)
    #Make prompt and retrieved nodes red
    fig.for_each_trace(lambda t: t.update(marker=dict(size=7, color='red', symbol='diamond')) if 'retrieved' in t['legendgroup'] else t)
    fig.for_each_trace(lambda t: t.update(marker=dict(size=12, color='red', symbol='diamond')) if 'prompt' in t['legendgroup'] else t)

    return st.plotly_chart(fig, use_container_width=False)