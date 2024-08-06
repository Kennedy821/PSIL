# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 17:43:24 2024

@author: worldcontroller
"""

import streamlit as st
from PIL import Image

im = Image.open('slug_logo.png')
st.set_page_config(
    page_title="Hello",
    page_icon=im,
)

st.write(f"# Welcome to PSIL by Slug!")

st.sidebar.success("Select a demo above.")

st.markdown(
    """
Welcome to PSIL song recommender, your gateway to discovering new music!

🎶 Dive in and find tunes that match how you feel, what you like, and what you need.
Whether you want to kick off your day with energy, relax after work, or get the right song for a road trip, PSIL makes it easy.

Start searching now and find your next top song with just a few clicks. Enjoy your music!

PSIL is a research project built to encourage music exploration based on the music you like and less on what's popular.

We will be publishing some research comparing PSIL to some well-known song-recommendation algorithms in the coming months.
"""
)