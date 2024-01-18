import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tensorflow as tf
import random
import tflearn
import json
import pickle
from sklearn.metrics import precision_score, recall_score, f1_score

#it extracts the root words from the string ignoring the characters
stemmer = LancasterStemmer()

#reading the json file
with open("intents.json") as file:
    data = json.load(file)

# Create test data and ground truth labels from the intents
test_queries = []
ground_truth_tags = []
for intent in data["intents"]:
    test_queries.extend(intent["patterns"])
    ground_truth_tags.extend([intent["tag"]] * len(intent["patterns"]))

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    #data preprocessing
    words = [] #contains word in the pattern
    labels = [] #contains tags in the json file
    docs_x = [] #contains list of tokenized word of each entry in pattern
    docs_y = [] #contains tag associated with each pattern element

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds) #word in the pattern list
            docs_y.append(intent["tag"])#to indicate the tag of which the word in the list is a part of

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    #stem each word and removing duplicates to make track of how many words or vocabulary, our model is looking at
    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words))) # set removes duplicates, list converts it into list, sort sorts the list
    labels = sorted(labels)

    #now we are doing one-hot encoding to make the input for the neural network as it doesn't accept non-int data
    output = []
    training = []
    out_empty = [0 for _ in range(len(labels))]

    for x,doc in enumerate(docs_x): 
        bag = []

        #This part it takes each word list from the doc_x list. The list represent each word of the sentence. We stem each word of this list and then in the for loop, we check that if our predefined words list contains the item that is also in wrds1 list, then we store it as 1 in the bag variable. Otherwise, we store 0 at that position in the bag variable. In this way, we one-hot encode each word from our training data set
        wrds1 = [stemmer.stem(w) for w in doc]

        for w in words:
            if w in wrds1:
                bag.append(1)
            else:
                bag.append(0)
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1 #looks for the index in docs_x and get the tag for that in doc_y then for that tag in labels list, set 1 at that index in that index of output list

        training.append(bag) #appends the one-hot encoded list of word to the list training.
        output.append(output_row) #appends the one-hot encoded list of labels for that pattern to the output list

    training = np.array(training)
    output = np.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output),f)

#resetting the default graph
tf.compat.v1.reset_default_graph()

# neural network
net = tflearn.input_data(shape=[None, len(training[0])]) # input layer 
net = tflearn.fully_connected(net, 8) # hidden layer
net = tflearn.fully_connected(net, 8) # hidden layer
net = tflearn.fully_connected(net,len(output[0]), activation="softmax") # output layer with softmax function to output the probabilities (between 0 and 1)
net = tflearn.regression(net) #defining the neural network as the regression model (logistic)

model = tflearn.DNN(net) #putting the neural network together
model.fit(training, output, n_epoch=2000, batch_size=8, show_metric=True) #training the model

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

#model eval---------------------------------------------------------------
#function to predict tags for test queries (for the evaluation of the model)
def predict_tags(test_queries):
    predicted_tags = []
    for query in test_queries:
        result = model.predict([user_One_Hot_Encoding(query, words)])
        tag = labels[np.argmax(result)]
        predicted_tags.append(tag)
    return predicted_tags
# Get predicted tags for test queries
predicted_tags = predict_tags(test_queries)
# Calculate precision, recall, and F1 score
precision = precision_score(ground_truth_tags, predicted_tags, average='weighted')
recall = recall_score(ground_truth_tags, predicted_tags, average='weighted')
f1 = f1_score(ground_truth_tags, predicted_tags, average='weighted')
#Print evaluation results
print("\n-------------EVALUATION--------------")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print("\n")
#------------------------------------------------------------------------------

#function for the actual model working
def chat():
    print("Start talking! (-1 to exit)")
    while True:
        inp = input("USER: ")
        if inp == "-1":
            break
        #Now we need to convert whatever the user has entered to the one-hot encoding format to feed to the model
        #the following line of code returns the list of probabilities i.e., the probability corresponding to each tag from our data set. We have to select the response with the highest probability
        result = model.predict([user_One_Hot_Encoding(inp, words)])

        #argmax function returns the index of max value in an np array
        tag = labels[np.argmax(result)]
    
        #now we randomly select any response from the dataset by traversing through the "intents" key and then through tag, if the tag matches with our predicted tag, then we randomly select the response from the responses list corresponding to that particular tag
        for tg in data["intents"]:
            if tg["tag"] == tag:
                responses = tg["responses"]
        print(f"CHATBOT: {random.choice(responses)}")

chat()