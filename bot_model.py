import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy as np
import tflearn
import tensorflow as tf
import random

import json
with open('intents.json') as json_data:
    intents = json.load(json_data)

words = []
classes = []
documents = []
ignore_words = ['?']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        # splits the input sentence into an array of words
        words_in_sentence = nltk.word_tokenize(str(pattern))
        words.extend(words_in_sentence)
        documents.append((words_in_sentence, str(intent['tag'])))

        if str(intent['tag']) not in classes:
            classes.append(str(intent['tag']))

# gets stems of all the words we've seen from the inputs
words = [stemmer.stem(word.lower()) for word in words if word not in ignore_words]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

print (len(documents), "documents")
print (len(classes), "classes", classes)
print (len(words), "unique stemmed words", words)

# convert our training inputs into vectors that show what words were in the sentence
# generate an output vector to classify the input as one of our conversation classes
training = []
output = []

output_empty = [0] * len(classes)
for doc in documents:
    words_vector = []
    #list of the words in this sentence
    sentence = doc[0]
    sentence = [stemmer.stem(word.lower()) for word in sentence]
    # creates a vector with 0 or 1 for every word we've seen
    for w in words:
        words_vector.append(1) if w in sentence else words_vector.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([words_vector, output_row])

random.shuffle(training)
training = np.array(training)

# training is a list of pairs of input words that correspond to a class
# separate list into x and y, by getting the input words in one list and
# corresponding class in another list
train_x = list(training[:,0])
train_y = list(training[:,1])

tf.reset_default_graph()

# deep learning library built on top of tensorflow to make creation of
# graphs simpler with a high level api.
# simple usage here: defining the input layer and the following layers of the
# network by showing what is coming in and the number of nodes that will be in the layer
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
# runs linear regression on the network, optimizing the loss function
# with a bunch of default characteristics (can be fiddled with)
net = tflearn.regression(net)

# define the type of network we're using and the examples to train on
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
# n_epch is number of iterations, batch_size is group of calculated errors to use in backprop
model.fit(train_x, train_y, n_epoch=1000, batch_size=8, show_metric=True)
model.save('model.tflearn')

import pickle
pickle.dump( {'words': words, 'classes': classes, 'train_x': train_x,
             'train_y': train_y}, open ("training_data", "wb") )
