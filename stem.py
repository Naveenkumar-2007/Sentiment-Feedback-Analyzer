import streamlit as st  
from nltk.stem import WordNetLemmatizer
import re
import pickle
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')  # Optional, but helps with WordNet lemmatization
def get_star_review(review):
    try:
        rating=int(review.strip())
        if 1<=rating<=5:
            return "⭐" * rating + " ({}/5)".format(rating)
    except:
        return None
    return None

with open('predict/vector.pkl','rb') as f:
    vector=pickle.load(f)
with open('predict/model.pkl','rb') as f:
    model=pickle.load(f)
def clean_text(text):
    text=text.lower()
    text=re.sub('[^a-z A-Z 0-9]','',text)
    text=re.sub(r'https?://\S+|www\.\S+','',text)
    text=BeautifulSoup(text,'lxml').get_text()
    text="".join([word for word in text.split() if word not in stopwords.words('english')])
    text=" ".join([WordNetLemmatizer().lemmatize(word) for word in text.split()])
    return text

st.title('Predict your Reviews')
#st.set_page_config(page_title="Review Rating Predictor", page_icon="⭐", layout="centered")
review_input=st.text_area('Enter your text Review or 1-5 review')
clean=clean_text(review_input)
if st.button('predict'):
    if not review_input or review_input.strip()==" ":
      st.warning('Please enter your Review')
    else:
        clean=clean_text(review_input)
        if clean.strip()==" ":
            st.warning("your Review Became after empty ")
        else:
            star_input=get_star_review(review_input)
            #if star_input:
            is_ratind=star_input is not None
            is_valid_text=not review_input.strip().isdigit()
            if is_ratind or is_valid_text:
                st.markdown(f"userinput:{star_input if star_input else review_input}")
                clean=clean_text(review_input)
                if not clean:
                    st.warning('your input became empty')
                else:
        
                    vector_pre=vector.transform([clean])
                    model_pre=model.predict(vector_pre)
                    if model_pre[0]==1:
                        st.write('✅ Postive Review')
                        st.success('😊')
                        
                    else:
                        st.error("❌ **Negative Review** 😞")
            else:
                st.warning('only numbers 1 to 5 numbers  or valid text are allowed')

