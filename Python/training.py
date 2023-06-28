import random
import json
import pickle
import numpy as np

import nltk
#nltk.download('punkt')
#nltk.download('wordnet')
#import nltk.download('punkt') had to use this to download punkt in order to work
#import nltk.download('wordnet') had to use this to download wordnet in order to work

from nltk.stem import WordNetLemmatizer

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD

#take the words and return base values eg: worked, works, work, working ->work
lemmatizer = WordNetLemmatizer()

#now we load the .json file, in here json file is read and load into load function and .json file is loaded
intents = json.loads(open('intents.json').read())
#this .json file acts as a dictionary in python

#3 emptylists
words = []
classes = []
documents = []

ignore_letters = ['?', '!', '.', ',']

#in here we access key values of intents object in intents.json
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        #this tolkenize sentences eg:("I am Eronne" -> "I", "am", "Eronne")
        words.extend(word_list)
        #these words tolkenized from sentence are now added to word list variable
        documents.append((word_list, intent['tag']))
        #to recognize these words are from this particular tag
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

#print(documents)

words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))
#in set use to eliminate duplicates while sort is used to sort the words in the list

#print(words)

classes = sorted(set(classes))
#this step is not essential because there is a least chance of having a duplicate

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))
#in this step we are going to save them to a file, wb refereed to writing binaries

#From here we are going to machine learning part

#now we have lot of characters and words which cannot be used to feed in to neural network,
#neural network requires numerical values so these characters must be in that form
#to do that we are going to use bag of words
#by using these these esach words will be displayed either 0 or 1 depending on their availability in patterns

training = []
output_empty = [0] * len(classes)
#this is a template of 0s, we need as many as 0s here

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)

train_x = list(training[:, 0])
train_y = list(training[:, 1])

#above are preprocessors for building neural net work, below actual building of neural network

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]), ), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
model.compile(loss= 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])

hist = model.fit(np.array(train_x), np.array(train_y), epochs = 200, batch_size = 5, verbose =1)
model.save('chatbot_model.h5', hist)

#print("Done")
#Now it's done here
#print("hello")







