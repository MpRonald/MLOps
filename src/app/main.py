from flask import Flask, request, jsonify
from flask_basicauth import BasicAuth
from textblob import TextBlob
from sklearn.linear_model import LinearRegression
import pickle
import os

cols = ['tamanho', 'ano', 'garagem']
model = pickle.load(open('../../models/model.sav', 'rb'))

app = Flask('__name__')
app.config['BASIC_AUTH_USERNAME'] = os.environ.get('BASIC_AUTH_USERNAME')
app.config['BASIC_AUTH_PASSWORD'] = os.environ.get('BASIC_AUTH_PASSWORD')
basic_auth = BasicAuth(app)

@app.route('/')
def home():
    return "My first API."

@app.route('/sentiment/<subject>')
@basic_auth.required
def sentiment(subject):
    tb = TextBlob(subject)
    polarity = tb.sentiment.polarity
    return "Polarity: {}".format(polarity)

@app.route('/cotacao/', methods=['POST'])
@basic_auth.required
def cotacao():
    data = request.get_json()
    data_input = [data[col] for col in cols]
    preco = model.predict([data_input])
    return jsonify(preco=preco[0])

app.run(debug=True, host='0.0.0.0')