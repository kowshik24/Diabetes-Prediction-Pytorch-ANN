from model import ANN_Model
import torch
from sklearn.preprocessing import StandardScaler
import pickle
import pandas as pd
import numpy as np
from flask import Flask, request , jsonify,request,url_for,render_template

model = torch.load('diabetes.pt')
scaler = pickle.load(open('scaler.pkl','rb'))

app = Flask(__name__)


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')



@app.route('/predict', methods=['POST','GET'])
def predict():
    if request.method == 'POST':
        input_data = request.form.to_dict()  # form data is accessible through request.form
        input_data = {k: float(v) for k, v in input_data.items()}
        input_data = scaler.transform(np.array(list(input_data.values())).reshape(1, -1))
        input_data = torch.FloatTensor(input_data)
        with torch.no_grad():
            prediction = model(input_data).argmax().item()
            if prediction == 1:
                result = 'The Patient has diabetes'
            else:
                result = 'The Patient has no diabetes'
            return render_template('index.html', result=result)  # pass the result to the template
    return render_template('index.html')  # render the form when the method is GET

if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0" port=5000)
