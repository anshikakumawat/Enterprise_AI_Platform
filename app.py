import streamlit as st
import nlp_pipelines

st.write(nlp_pipelines.__file__)
from nlp_pipelines import analyze_text

st.title("NLP Pipeline Dashboard")

st.write("Enter text to perform NLP analysis")

text = st.text_area("Enter your text")

if st.button("Analyze"):
    if text:
        result = analyze_text(text)
        st.write(result)

        st.subheader("Results")

        st.write("### Tokens")
        st.write(result["Tokens"])

        st.write("### Filtered Tokens")
        st.write(result["Filtered Tokens"])

        st.write("### Stemmed Tokens")
        st.write(result["Stemmed Tokens"])

        st.write("### Lemmatized Tokens")
        st.write(result["Lemmatized Tokens"])

        st.write("### POS Tags")
        st.write(result["POS Tags"])

        st.write("### Named Entities")
        st.write(result["Named Entities"])

        st.write("### Sentiment")
        st.success(result["Sentiment"])

        st.write("### TF-IDF Features")
        st.write(result["TF-IDF"]["features"])

        st.write("### TF-IDF Values")
        st.write(result["TF-IDF"]["values"])

        st.write("### Word2Vec Model")
        st.write(result["Word2Vec"])

        st.write("### GloVe")
        # st.write(result["GloVe"])

        st.write("### Embedding Comparison")
        st.write(result["Comparison"])

    else:
        st.warning("Please enter some text")