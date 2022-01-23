import json
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
import colorama
colorama.init()
from colorama import Fore, Style, Back
import random
import pickle


with open('intents.json') as f:
    data = json.load(f)


def chat():
    #load saved model, tokenizer, label_encoder
    model = keras.models.load_model('chatbot_model')

    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    with open('label_encoder.pickle', 'rb') as handle:
        label_encoder = pickle.load(handle)

    while True:
        inp = input(Fore.YELLOW + "User: " + Style.RESET_ALL)
        if(inp.lower() == "quit"):
            break;
        
        result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([inp]), truncating='post', maxlen=20))
        tag = label_encoder.inverse_transform([np.argmax(result)])

        for intent in data['intents']:
            if intent['tag'] == tag:
                print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL, np.random.choice(intent['responses']))

print(Fore.YELLOW + "Start messaging with the chatbot, type quit to stop!" + Style.RESET_ALL)
chat()