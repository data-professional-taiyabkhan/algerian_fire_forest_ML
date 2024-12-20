import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

from flask import Flask, render_template, request

app = Flask(__name__)

ridge_model = pickle.load(open('model/ridge.pkl','rb'))
StandardScaler = pickle.load(open('model/scalar.pkl','rb'))

@app.route("/")
def helloworld():
    return render_template("index.html")

@app.route("/predictdata", methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        Temprature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        data = StandardScaler.transform([[Temprature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result = ridge_model.predict(data)
        return render_template("predictdata.html",result = result[0])
    else:
        return render_template("predictdata.html")

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)