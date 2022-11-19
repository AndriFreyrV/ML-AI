from flask import Flask, request
from flask_restful import Resource, Api
import pickle
from sklearn.ensemble import RandomForestRegressor
import requests
import os

app = Flask(__name__)
api = Api(app)

model=pickle.load(open('model.pk1','rb+'))


def predict(features):
    return model.predict([features])

def add_one_hot_vars(params):
    seasons = {f'season_{i}':1 if params['season'] == i else 0 for i in range(1,5)}
    params.update(seasons)
    params.pop('season', None)
    weathersits = {f'weathersit_{i}':1 if params['weathersit'] == i else 0 for i in range(1,5)}
    params.update(weathersits)
    params.pop('weathersit', None)
    return params

@app.route('/test')
def test():
    # test calling the api with some params
    PARAMS = {
        'yr': 0.0,
        'mnth': 1.0,
        'hr': 1.0,
        'weekday': 6.0,
        'workingday': 0.0,
        'temp': 0.22,
        'atemp': 0.2727,
        'hum': 0.8,
        'windspeed': 0.0,
        'season': 1.0,
        'weathersit': 1.0,
    }
    r = requests.get('http://ec2-18-232-63-74.compute-1.amazonaws.com:8080/predict', params = PARAMS)
    return r.json()


class pred(Resource):
    def get(self):
        args = request.args
        params = add_one_hot_vars(dict(args))
        features = []
        for k, v in params.items():
            features.append(float(v))
        pred = predict(features)
        return {'pred':pred[0]}

api.add_resource(pred, '/predict')

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=8080)