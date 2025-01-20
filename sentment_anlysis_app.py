import streamlit as st
import pickle
import helper
# Set up the text input and button
st.title("sentmint analysis app")

lr_model = pickle.load(open("artifacts/lr.pkl","rb"))
# Input text box
review = st.text_input("Enter your review here:")

# Button
if st.button("result"):
    if review:  
        processed_review = helper.preproccessing(review).toarray()
        pred = lr_model.predict(processed_review)
        if   pred == 1 :
              st.success(f"this is positive review")
        else:
              st.success(f"this is negative review")
    else:
        st.warning("Please enter review again.")




