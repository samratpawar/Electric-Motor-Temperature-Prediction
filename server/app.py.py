from flask import Flask, jsonify, render_template, request
import joblib
import sklearn
import os
import numpy as np


app = Flask(__name__)


@app.route("/")
def index():
    return render_template("home.html")

@app.route('/predict',methods=['POST','GET'])
def result():

    ambient= float(request.form['ambient'])
    coolant=float(request.form['coolant'])
    u_d= float(request.form['u_d'])
    u_q= float(request.form['u_q'])
    motor_speed = float(request.form['motor_speed'])
    torque= float(request.form['torque'])
    i_d= float(request.form['i_d'])


    model_path = r"C:\Flask\models\KNNRegressor.sav"

    model = joblib.load(model_path)

    result = (np.array([ambient, coolant, u_d, u_q, motor_speed, torque, i_d]).reshape(1, -1))

    prediction = model.predict(result)


    return render_template('index.html', result=prediction)


if __name__ == "__main__":
    app.run(debug=True, port=9457)