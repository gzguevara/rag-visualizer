import streamlit as st

import joblib
import pandas as pd

import umap

from transformers import AutoTokenizer, AutoModelForCausalLM

from llama_index import (
    StorageContext,
    ServiceContext,
    load_index_from_storage,
    download_loader,
    VectorStoreIndex,
)

from transformers import AutoTokenizer, AutoModelForCausalLM
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.prompts.prompts import SimpleInputPrompt
from llama_index.llms import HuggingFaceLLM

from constants import (
    SYSTEM_PROMPT
)


model_name  = "meta-llama/Llama-2-7b-chat-hf"
auth_token  = "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"
emb_name    = "sentence-transformers/all-MiniLM-L6-v2"
storage_dir = "/home/ubuntu/chatbot/media/storage"
reducer_dir = '/home/ubuntu/chatbot/dev_notebooks/umap_reducer.joblib'
pdf_dir     = '/home/ubuntu/chatbot/media/Accenture-Fiscal-2023-Annual-Report.pdf'

@st.cache_resource
def get_service_context():

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=auth_token,
        )

    # Create model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=auth_token,
        load_in_8bit=True,
        )

    # Create and dl embeddings instance
    embed_model=HuggingFaceEmbedding(
        model_name=emb_name,
        )

    # Create a HF LLM using the llama index wrapper
    llm = HuggingFaceLLM(
        context_window       = 4096,
        max_new_tokens       = 256,
        system_prompt        = SYSTEM_PROMPT,
        query_wrapper_prompt = SimpleInputPrompt("{query_str} [/INST]"),
        model                = model,
        tokenizer            = tokenizer,
        )

    # Create new service context instance
    service_context = ServiceContext.from_defaults(
        llm           = llm,
        embed_model   = embed_model,
        chunk_overlap = 20,
        chunk_size    = 256,
    )

    return service_context

@st.cache_resource
def make_index():

    loader = download_loader("PDFReader")
    loader = loader()

    documents = loader.load_data(file=pdf_dir)

    for doc in documents:

        doc.text_template = '{content}'

    index = VectorStoreIndex.from_documents(documents)

    st.session_state["index"] = index

@st.cache_resource
def get_query_engine(num_chuncks):

    index = st.session_state["index"]
    engine = index.as_query_engine(similarity_top_k=num_chuncks)

    return engine

@st.cache_resource
def make_reducer():

    print('making reducer...')

    index = st.session_state["index"]

    idx=[]
    embs=[]

    for key, emb in index._vector_store._data.embedding_dict.items():

        idx.append(key)
        embs.append(emb)

    reducer = umap.UMAP(n_components=3, random_state=23)
    embs = reducer.fit_transform(embs)
    embs = pd.DataFrame(embs, index=idx, columns=['col0', 'col1', 'col2'])

    st.session_state["reducer"] = reducer
    st.session_state["embeddings_reduced"] = embs
