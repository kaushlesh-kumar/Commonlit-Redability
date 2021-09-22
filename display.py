import streamlit as st
import pandas as pd
import numpy as np
import src.predict_score

st.write("""
# Assessing the readability of a text piece

This app provides a simple assessment of the readability of your text
""")

user_input = st.text_area("Enter the text piece")

print(type(user_input))

test_df= pd.DataFrame([user_input], columns=["excerpt"])

score = predict_score.get_score(test_df)

if score:
    #st.text("The readability score of the text above is")
    st.metric("The readability score of the text above is", str(round(score, 2)))
    #print(score)


