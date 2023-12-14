import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer as ps
import string

nltk.download('stopwords')

#lets load the saved vectorizer and nave bayes model
tfidf = pickle.load(open('C:\\Users\\nithin sudheer\\my vs codes\\spam detection\\vectorizer.pkl','rb'))
model = pickle.load(open('C:\\Users\\nithin sudheer\\my vs codes\\spam detection\\model.pkl','rb'))

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    text = [word for word in text if word.ialnum()]

    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]

    text = [ps.stem(word) for word in text]

    return " ".join(text)

#saving streamlit code
st.title("email spam classifier")
input_sms = st.text_area("enter message")

if st.button('predict'):
    #preprocess
    transformed_sms = transform_text(input_sms)
    #vectorize
    vector_input = tfidf.transform([transformed_sms])
    #predict
    result = model.predict(vector_input)[0]
    #display
    if result ==1:
        st.header("spam")
    else:
        st.header("not spam")
