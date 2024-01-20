# Import streamlit for app dev
import streamlit as st

# Import transformer classes for generaiton
from transformers import AutoTokenizer, AutoModelForCausalLM
# Import torch for datatype attributes
import torch
# Import the prompt wrapper...but for llama index
from llama_index.prompts.prompts import SimpleInputPrompt
# Import the llama index HF Wrapper
from llama_index.llms import HuggingFaceLLM
# Bring in embeddings wrapper
from llama_index.embeddings import HuggingFaceEmbedding

# Bring in stuff to change service context
from llama_index import set_global_service_context
from llama_index import ServiceContext
# Import deps to load documents
from llama_index import VectorStoreIndex, download_loader
from pathlib import Path

# Define variable to hold llama2 weights naming
model_name = "meta-llama/Llama-2-7b-chat-hf"
# Set auth token variable from hugging face
auth_token = "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"

@st.cache_resource
def get_tokenizer_model():

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir='./model/',
        token=auth_token,
        )

    # Create model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir='./model/',
        token=auth_token,
        load_in_8bit=True,
        )

    return tokenizer, model

tokenizer, model = get_tokenizer_model()

# Create a system prompt
system_prompt = """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as
helpfully as possible, while being safe. Your answers should not include
any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.
Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain
why instead of answering something not correct. If you don't know the answer
to a question, please don't share false information.

Your goal is to provide answers relating to the financial performance of
the company.<</SYS>>
"""
# Throw together the query wrapper
query_wrapper_prompt = SimpleInputPrompt("{query_str} [/INST]")

# Create a HF LLM using the llama index wrapper
llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=256,
    system_prompt=system_prompt,
    query_wrapper_prompt=query_wrapper_prompt,
    model=model,
    tokenizer=tokenizer,
    )

# Create and dl embeddings instance
embed_model=HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create new service context instance
service_context = ServiceContext.from_defaults(
    chunk_size=512,
    llm=llm,
    embed_model=embed_model
)
# And set the service context
set_global_service_context(service_context)

# Download PDF Loader
PDFReader = download_loader("PDFReader")
# Create PDF Loader
loader = PDFReader()
# Load documents
documents = loader.load_data(file=Path('/home/ubuntu/chatbot/media/Accenture-Fiscal-2023-Annual-Report.pdf'))

# Create an index - we'll be able to query this in a sec
index = VectorStoreIndex.from_documents(documents)
# Setup index query engine using LLM
query_engine = index.as_query_engine()

# Create centered main title
st.title('🦙 Llama Banker')
# Create a text input box for the user
prompt = st.text_input('Input your prompt here')

# If the user hits enter
if prompt:
    response = query_engine.query(prompt)
    # ...and write it out to the screen
    st.write(response.response)

    # Display raw response object
    with st.expander('Response Object'):
        st.write(response)
    # Display source text
    with st.expander('Source Text'):
        st.write(response.get_formatted_sources())


