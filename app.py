import pickle
import numpy as np
from flask import Flask,request,render_template

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_feature = [np.array(int_features)]
    prediction = model.predict(final_feature)

    return render_template('home.html',prediction_text='Predicted Class : {}'.format(prediction))

if __name__=='__main__':
    app.run(debug=True)

