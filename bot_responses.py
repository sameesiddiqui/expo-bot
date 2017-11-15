# reload our model, weights, and data

import pickle
data = pickle.load( open("training_data", "rb") )
words = data['words']
classes = data['classes']
train_x = data['train_x']
train_y = data['train_y']

import json
with open('intents.json') as json_data:
    intents = json.load(json_data)

import tflearn
import tensorflow as tf

tf.reset_default_graph()

net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
model.load('./model.tflearn')

import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

# get word stems for an input sentence
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# have we seen this word stem before? if so, set the value in our seenWords vector to 1
# we now have a way to take any sentence and convert it to the input format that we accept in our NN
def generate_words_vector(sentence, words, show_details=False):
    sentence_words = clean_up_sentence(sentence)
    seen_words = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if (w == s):
                seen_words[i] = 1
                if show_details:
                    print ("found in seen_words: %s" % w)
    return seen_words

ERROR_THRESHOLD = 0.25
context = {}

import random
def classify(sentence):
    # given this sentence, turn it into a vector with what
    # words are present, and use the model to predict
    # its category
    results = model.predict([generate_words_vector(sentence, words)])[0]

    # get sorted list of predictions - from highest to lowest confidence
    results = [[i, r] for i,r in enumerate(results) if r > ERROR_THRESHOLD]
    results.sort(key = lambda x: x[1], reverse=True)
    return_list = []

    # create a list of probabilities for the intent
    # of the sentence
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    return return_list

def response(sentence, userID='123', show_details=False):
    # get category of the sentence
    results = classify(sentence)
    print (results)
    if results:
        while results:
            for i in intents['intents']:
                if i['tag'] == results[0][0]:
                    # TODO: do the same for fallback error
                    # handle multi line strings in json file
                    if i['tag'] == 'questions':
                        return str(i['responses'].join('\n'))

                    # if this is a conversation path that will have context, we need to store the info
                    if 'context_set' in i:
                        if show_details: print ('context:', i['context_set'])
                        context[userID] = i['context_set']

                    # if we get a match with 'context_filter' only return its responses if we're in that context
                    # if we don't need to filter for context, return one of the responses
                    if not 'context_filter' in i or \
                        (userID in context and 'context_filter' in i and i['context_filter'] == context[userID]):
                        if show_details: print ('tag:', i['tag'])
                        return str(random.choice(i['responses']))

            results.pop(0)
