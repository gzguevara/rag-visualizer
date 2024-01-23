import streamlit as st
import umap
import plotly.express as px
import pandas as pd
import numpy as np

def plot_embeddings(prompt, response, context):

    embs = st.session_state["embeddings_reduced"]

    if prompt and response:

        reducer = st.session_state["reducer"]
        prompt_emb = context.embed_model.get_agg_embedding_from_queries([prompt])
        prompt_emb = reducer.transform([prompt_emb])
        prompt_emb = pd.DataFrame(prompt_emb, index=['prompt'], columns=['col0', 'col1', 'col2'])

        embs = pd.concat([embs, prompt_emb])

        nodes = [node.id_ for node in response.source_nodes]
        embs['class'] = np.where(embs.index.isin(nodes), 'retrieved', 'node')
        embs.loc[embs.index=='prompt', 'class'] = 'prompt'

    fig = px.scatter_3d(
        embs,
        x='col0',
        y='col1',
        z='col2',
        color='class' if prompt else None,
        hover_name=embs.index,
        color_discrete_sequence=['blue', 'green', 'red'],
        )

    fig.update_traces(marker_size=3)
    fig.update_layout(width=850, height=750)
    fig.update_traces(showlegend=False)

    return st.plotly_chart(fig, use_container_width=False)