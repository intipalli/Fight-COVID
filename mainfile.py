import json
import nltk
import pickle
import random
import numpy as np

from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()

from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
from keras.optimizers import SGD

with open(r"intents.json","r") as file:
    data=json.load(file)
    
twords=[]
labels=[]
docs=[]

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds=nltk.word_tokenize(pattern)
        twords.extend(wrds)
        docs.append((wrds,intent["tag"]))
        
    if intent["tag"] not in labels:
        labels.append(intent["tag"])

ignore=["?",",","!"]
twords=[lemmatizer.lemmatize(w.lower()) for w in twords if w not in ignore]
twords=sorted(list(set(twords)))
labels=sorted(list(set(labels)))

pickle.dump(twords,open(r"twords.pkl","wb"))
pickle.dump(labels,open(r"labels.pkl","wb"))

training=[]
output=[0]*len(labels)
for d in docs:
    bag=[]
    pattern_words=d[0]
    pattern_words=[lemmatizer.lemmatize(wrd.lower()) for wrd in pattern_words]
    
    for w in twords:
        if w in pattern_words:
            bag.append(1)
        else:
            bag.append(0)
    
    output_row=list(output)
    output_row[labels.index(d[1])]=1
    
    training.append([bag,output_row])

random.shuffle(training)
training=np.array(training)
train_pattern=list(training[:,0])
train_intent=list(training[:,1])
print("Training data created")

model = Sequential()
model.add(Dense(128, input_shape=(len(train_pattern[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_intent[0]), activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

hist = model.fit(np.array(train_pattern), np.array(train_intent), epochs=200, batch_size=5, verbose=1)
model.save(r"chatbot_model.h5", hist)

print("model created")
