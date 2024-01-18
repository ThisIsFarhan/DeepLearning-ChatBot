import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tensorflow as tf
import random
import tflearn
import json
import pickle
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware


#it extracts the root words from the string ignoring the characters
stemmer = LancasterStemmer()

#reading the json file
with open("intents.json") as file:
    data = json.load(file)

#reading the data file
with open("data.pickle", "rb") as f:
    words, labels, training, output = pickle.load(f)

#resetting the default graph
tf.compat.v1.reset_default_graph()

# neural network
net = tflearn.input_data(shape=[None, len(training[0])]) # input layer 
net = tflearn.fully_connected(net, 8) # hidden layer
net = tflearn.fully_connected(net, 8) # hidden layer
net = tflearn.fully_connected(net,len(output[0]), activation="softmax") # output layer with softmax function to output the probabilities (between 0 and 1)
net = tflearn.regression(net) #defining the neural network as the regression model (logistic)
model = tflearn.DNN(net) #putting the neural network together
model.load("model.tflearn")

#function to make the input string in the form of one-hot encoding
def user_One_Hot_Encoding(s, wordslist):
    bag = [0 for _ in range(len(wordslist))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for x in s_words:
        for i,w in enumerate(wordslist):
            if w == x:
                bag[i] = 1
    return np.array(bag)

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class model_input(BaseModel):
    instring : str

@app.post('/predict_response')
def myPrediction(inputData:model_input):
    myJsonInput = inputData.json()
    inputDict = json.loads(myJsonInput)
    
    query = inputDict["instring"]

    result = model.predict([user_One_Hot_Encoding(query, words)])
    tag = labels[np.argmax(result)]

    for tg in data["intents"]:
            if tg["tag"] == tag:
                responses = tg["responses"]
    resp = random.choice(responses)
    return {"Response": resp}



