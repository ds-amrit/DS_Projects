# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 20:20:56 2022

@author: amrit
"""


# IMPORTING LIBRARIES
import streamlit as st
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

st.write("# Real Time Sentiment Analysis")

# Accepting user input
user_input = st.text_input("Enter text to analyze it's sentiment >>: ")

# Downloading VADER LEXICON
nltk.download("vader_lexicon")

# Initializing Sentiment Analysis Model
sia = SentimentIntensityAnalyzer()

# Calculating the sentiment score for the input
score = sia.polarity_scores(user_input)

if score == 0:
    st.write(" ")
elif score["neg"] != 0:
    st.write("# Negative")
elif score["pos"] != 0:
    st.write("# Positive")
