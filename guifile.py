import os
import sys
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import keras
sys.stderr = stderr


import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np

import tkinter
from tkinter import *
import json
import random

from keras.models import load_model
model = load_model(r"chatbot_model.h5")

intents = json.loads(open(r"intents.json").read())
words = pickle.load(open(r"twords.pkl",'rb'))
classes = pickle.load(open(r"labels.pkl",'rb'))

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res


def send():
    message = sentence.get("1.0","end-1c",).strip()
    sentence.delete("0.0",END)

    if message != '':
        chat.config(state=NORMAL)
        chat.insert(END, "User: " + message + '\n\n')
        chat.config(fg="#442265", font=("Arial", 12))
        res = chatbot_response(message)
        chat.insert(END, "COVID-19 agent: " + res + '\n\n')
        chat.config(state=DISABLED)
        chat.yview(END)
window = Tk()
window.title("COVID-19 - Prevention is better than CURE");
window.geometry("400x465");


chat = Text(window, bd=0, bg="white", height="8", width="50", font="Arial")

scrolling = Scrollbar(window, command=chat.yview, cursor="heart")
photo = PhotoImage(file = r"index1.png")

button = Button(window,image=photo,compound = LEFT,command=send)

sentence = Text(window, bd=0, bg="white", width="29", height="5", font="Arial")

scrolling.place(x=376, y=6, height=386)
chat.place(x=6, y=6, height=386, width=370)
sentence.place(x=6, y=401, height=60, width=315)
button.place(x=316, y=401,height=60,width=80)

window.mainloop()
