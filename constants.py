PLOT_SIZE = 3

ABOUT_THIS_APP = "This application is inspired and adapts the code from [this excellent course](https://www.deeplearning.ai/short-courses/advanced-retrieval-for-ai/) by DeepLearning.AI and Chroma."

CHUNK_EXPLAINER = "In RAG, your document is divided into parts (i.e. chunks) for searching, and relevant chunks are given to the LLM as additional context. \n \n"\
                "\"Chunk Size\" is the number of tokens in one of these chunks, and \"Chunk Overlap\" is the number of tokens shared between consecutive chunks to maintain context. \n\n"\
                "One word is about 3-4 tokens."

BUILD_VDB_LOADING_MSG = 'Building the vector database 🚧 ... It takes about 1-2 minutes, depending on the size of your PDF. Please do not close this app. 🙏'

VISUALISE_LOADING_MSG = 'Visualising your chunks 🎨 ... It takes about 2-3 minutes, depending on the size of your PDF. Please do not close this app. 🙏'

VISUALISATION_SETTINGS = {
            'Original Query': {'color': 'red', 'opacity': 1, 'symbol': 'cross', 'size': 15},
            'Retrived': {'color': 'green', 'opacity': 1, 'symbol': 'circle', 'size': 10},
            'Chunks': {'color': 'blue', 'opacity': 0.4, 'symbol': 'circle', 'size': 10},
            'Sub-Questions': {'color': 'purple', 'opacity': 1, 'symbol': 'star', 'size': 15},
            'Hypothetical Ans': {'color': 'purple', 'opacity': 1, 'symbol': 'star', 'size': 15},
        }

SYSTEM_PROMPT = """<s>[INST] <<SYS>>
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
