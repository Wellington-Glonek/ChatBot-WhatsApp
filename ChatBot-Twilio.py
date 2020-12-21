"""
Created on Mon Dec 21 15:20:57 2020

@author: wellington
"""
from flask import Flask, request
import requests
from twilio.twiml.messaging_response import MessagingResponse

app = Flask(__name__)


# Importar pacotes necessários
import nltk
#nltk.download('punkt')
#from nltk.stem.lancaster import LancasterStemmer
from tensorflow.python.framework import ops
# stemmer = LancasterStemmer()
# nltk.download('rslp')
# Biblioteca para realizar stemizaçao das palavras
stemmer = nltk.stem.RSLPStemmer()

# Biblioteca para remover palavras conectores ("a", "o", "aos", "as", etc.)
nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('portuguese')

# from sklearn.feature_extraction.text import TfidfVectorizer

# Biblioteca para retirar acentuação
import unidecode

from datetime import datetime
now = datetime.now()
now = now.strftime('%m/%d/%Y %H:%M')

import numpy
import tflearn
import tensorflow
import random
import json
import pickle

# Abrir JSON com perguntas e respostas do ChatBot
with open("intents.json") as file:
    data = json.load(file)

# Carregar base salva caso exista. O arquivo data.pickle contem as informações treinadas do intentes.json e desta forma abre mais rápido. 
# Caso queira treinar novamente com novas informações do arquuvo intentes.json, exclua o arquivo data.pickle e rode o script novamente
try:
	with open("data.pickle", "rb") as f:
		words, labels, training, output = pickle.load(f)
except:
	words = []
	labels = []
	docs_x = []
	docs_y = []

# Separacao das palavras
	for intent in data["intents"]:
		for pattern in intent["patterns"]:
			wrds = nltk.word_tokenize(pattern)
			words.extend(wrds)
			docs_x.append(wrds)
			docs_y.append(intent["tag"])

			if intent["tag"] not in labels:
				labels.append(intent["tag"])

# Fazer tratamento das palavras (remover stopwords e stemização)
# Ainda estou testando as stopwords, pois algumas questões simples não são entendidas por conta das stopwords. 
# Ex. O que e voce? Neste caso temos muitos conectores "o", "que", "e". Ao remover obtemos apenas "voce", dificultando o entendimento da Luna
# Caso o uso de stopwords prejudique o entendimento da Luna, retirar a linha contendo "and w not in stopwords" e deixar a linha acima que esta comentada.
	words = [stemmer.stem(w.lower()) for w in words if w != "?"]
#	words = [stemmer.stem(w.lower()) for w in words if w != "?" and w not in stopwords]
	words = sorted(list(set(words)))

	labels = sorted(labels)

	training = []
	output = []

	out_empty = [0 for _ in range(len(labels))]

# Criar saco de palavras
	for x, doc in enumerate(docs_x):
		bag = []

		wrds = [stemmer.stem(w) for w in doc]

		for w in words:
			if w in wrds:
				bag.append(1)
			else:
				bag.append(0)

		output_row = out_empty[:]
		output_row[labels.index(docs_y[x])] = 1

		training.append(bag)
		output.append(output_row)

	training = numpy.array(training)
	output = numpy.array(output)

# Este arquivo salva as informações do arquivo "intentes.json". 
with open("data.pickle", "wb") as f:
	pickle.dump((words, labels, training, output), f)


ops.reset_default_graph()

# Criando rede neural para definir respostas para cada pergunta
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

# Caso o modelo já esteja treinado, abrir "model.tflearn" caso contrario treinar o modelo 
# Por algum motivo não está funcionando. Quando o modelo não foi treinado (Arquivo "model.tflearn não existe"), ao inves de entrar na linha de exception e treinar o modelo o sistem emite erro
#try: 
#	model.load("model.tflearn")
#except:
#	model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
#	model.save("model.tflearn")

# Visto que o código acima não funciona, descomentar a linha abaixo para rodar o script já treinado
#model.load("model.tflearn")

# Visto que o código acima não funciona, descomentar as duas linhas abaixo para treinar e salvar o modelo. 
# Desta forma na proxima vez que rodar não precisa treinar o modelo novamente.
model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
model.save("model.tflearn")


def bag_of_words(s, words):
	bag = [0 for _ in range(len(words))]

	s_words = nltk.word_tokenize(s)
	s_words = [stemmer.stem(word.lower()) for word in s_words]

	for se in s_words:
		for i, w in enumerate(words):
			if w == se:
				bag[i] = 1
	return numpy.array(bag)

@app.route('/bot', methods=['POST'])
# Código para o Chat
def bot():
    inp = request.values.get('Body', '').lower()
    results = model.predict([bag_of_words(inp, words)])[0]
    results_index = numpy.argmax(results)
    tag = labels[results_index]
    resp = MessagingResponse()
    msg = resp.message()

    if results[results_index] > 0.7:
        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']
        responses = random.choice(responses)
        msg.body(responses)

    	# Gravar log da conversa
        arquivo = open("log_chat.txt", "a")
        frases = now + ": Você: "+inp+ "\n" + now +": Luna: " + responses + "\n"
        arquivo.writelines(frases)
        arquivo.close()

    else:
        msg.body('Não entendi, poderia falar novamente em outras palavras?')
			
		# Gravar log de erros e log da conversa
        arquivo = open("log_error.txt", "a")
        frases = now + ": Você: " + inp + "\n" + now + ": Luna: Não entendi, poderia falar novamente em outras palavras? \n"
        arquivo.writelines(frases)
        arquivo = open("log_chat.txt", "a")
        frases = now + ": Você: " + inp + "\n" + now + ": Luna: Não entendi, poderia falar novamente em outras palavras? \n"
        arquivo.writelines(frases)
        arquivo.close()

    return str(resp)

if __name__ == '__main__':
   app.run()
