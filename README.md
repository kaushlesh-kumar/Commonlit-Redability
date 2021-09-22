# Commonlit-Redability - A classical machine learning approach

This is a classical machine learning approach to solve the [CommonLit-Readability Prize challenge](https://www.kaggle.com/c/commonlitreadabilityprize).
While the winner entries used ensamble of tranformer based models to acieve the winning score, this approach take in the domain based approach to create hypotheses and features which are then used to train classicla  ML models to predict the scores. 

AutoML has been used to reduce the experimentation time and put more focus on creating features. [PyCaret](https://github.com/pycaret/pycaret) used to automate the ML workflow.

To implement use a `python 3.8.10` environment. The hard dependecies ar listed in `requirements.txt`.

This code also implements a [Streamlit](https://streamlit.io/) based simple UI. To launch the UI navigate to the root folder and run
`streamlit run display.py`
