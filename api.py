from fastapi import FastAPI
import pickle 
from pydantic import BaseModel
app=FastAPI()
with open('predict/vector.pkl','rb') as f:
    vector=pickle.load(f)
with open('predict/model.pkl','rb') as f:
    model=pickle.load(f)

class Base_model(BaseModel):
    reviewText:str

def clean_text(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z0-9 ]+', '', text)  
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = BeautifulSoup(text, 'lxml').get_text()
    text = " ".join([word for word in text.split() if word not in stopwords.words('english')])
    text = " ".join([WordNetLemmatizer().lemmatize(word) for word in text.split()])
    return text.strip()


@app.post('/predict')
def prediction(item:Base_model):
    review=clean_text([item.reviewText])
    vector_pre=vector.transform([review])
    model_pre=model.predict(vector_pre)
    sentiment = "Positive 😀" if model_pre[0] == 1 else "Negative 😞"
    return {
        "review": item.reviewText,
        "prediction": int(model_pre[0]),   
        "sentiment": sentiment
    }
