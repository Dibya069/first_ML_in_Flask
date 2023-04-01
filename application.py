from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

applicatoin = Flask(__name__)
app = applicatoin

## Importing scaler regression and ridge regression
ridge_model = pickle.load(open("mdoels/ridge.pkl", "rb"))
scale_model = pickle.load(open("mdoels/scale.pkl", "rb"))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predictdata", methods = ["GET", "POST"])
def predict_FWI():
    if request.method == "POST":
        Temperature = float(request.form.get("Temperature"))
        RH = float(request.form.get("RH"))
        Ws = float(request.form.get("Ws"))
        Rain = float(request.form.get("Rain"))
        FFMC = float(request.form.get("FFMC"))
        DMC = float(request.form.get("DMC"))
        ISI = float(request.form.get("ISI"))
        Classes = float(request.form.get("Classes"))
        Region = float(request.form.get("Region"))

        new_scale_data = scale_model.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result = ridge_model.predict(new_scale_data)

        return render_template("home.html", results = result[0])

    else:
        return render_template("home.html")


if __name__=="__main__":
    app.run(host="0.0.0.0")
