import streamlit as st

import joblib
import pandas as pd

from transformers import AutoTokenizer, AutoModelForCausalLM

from llama_index import StorageContext, ServiceContext, load_index_from_storage
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
storage_dir = "/home/ubuntu/chatbot/dev_notebooks/storage"
reducer_dir = '/home/ubuntu/chatbot/dev_notebooks/umap_reducer.joblib'

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
        llm         = llm,
        embed_model = embed_model,
        )

    return service_context

@st.cache_resource
def get_query_engine(num_chuncks):

    context = StorageContext.from_defaults(persist_dir=storage_dir)
    index   = load_index_from_storage(context)
    engine  = index.as_query_engine(similarity_top_k=num_chuncks)

    return engine

@st.cache_resource
def get_reducer():

    return joblib.load(reducer_dir)

@st.cache_data
def get_emb_space():

    query_engine = get_query_engine()
    vec_dict = query_engine._retriever._vector_store._data.embedding_dict
    emb = [vec_dict[key] for key in vec_dict]
    emb = pd.DataFrame(emb, index=vec_dict.keys())

    return emb

@st.cache_data
def reduce_emb_space():

    reducer = get_reducer()
    emb_space = get_emb_space()
    emb_red = reducer.transform(emb_space.values)

    emb_red = pd.DataFrame(emb_red, index=emb_space.index, columns=[f'col_{i}' for i in range(3)])

    return emb_red