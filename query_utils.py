import streamlit as st

from model_utils import get_query_engine

def parse_query(prompt, num_chuncks):

    engine = get_query_engine(num_chuncks)

    if prompt:

        response = engine.query(prompt)

        box_height = min(300, 20 + len(response.response) // 5) if response.response else 100

        return response, box_height

    else:

        return None, None