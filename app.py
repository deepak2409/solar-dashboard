from flask import Flask, render_template, request, send_file
import joblib, os
from flask import flash, request, redirect, url_for
from werkzeug.utils import secure_filename

import numpy as np
import pandas as pd, xlrd, openpyxl


UPLOAD_FOLDER = 'C:\\Users\\HP Account\\Desktop\\Mobius-Internship\\Solar Analytics\\SolarApp_V2'
ALLOWED_EXTENSIONS = {'xlsx'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

#@app.route('/test')
#def test():
#    return "Flask is being used for development"

@app.route('/')
def wel():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/Bulk_Forecast', methods = ['GET', 'POST'])
def Bulk_Forecast():
    return render_template('Bulk_Forecast.html')

@app.route("/predict", methods = ['GET','POST'])
def predict():
    if request.method == 'POST':
        try:
            Temperature = float(request.form['Temperature'])
            RelativeHumidity= float(request.form['RelativeHumidity'])
            Pressure = float(request.form['Pressure'])
            Windspeed=float(request.form['Windspeed'])
            Winddirection= float(request.form['Winddirection'])
            Rainfall=float(request.form['Rainfall'])
            Shortwave = float(request.form['ShortWave'])
            pred_args = [Temperature,RelativeHumidity,Pressure,Windspeed,Winddirection,Rainfall,Shortwave]
            pred_args_arr = np.array(pred_args)
            pred_args_arr = pred_args_arr.reshape(1,-1)
            mul_reg = open("Madurai_model.pkl","rb")
            ml_model = joblib.load(mul_reg)
            model_prediction = ml_model.predict(pred_args_arr)
            model_prediction = [round(float(i),2) for i in model_prediction[0]]
            model_prediction = sum(model_prediction)
        except ValueError:
            return "Please check if the values are entered correctly "
    return render_template('predict.html', prediction = model_prediction)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/submit', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
        test_data = pd.read_excel(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
        new_df = pd.DataFrame(columns = ['Sno', 'Date', 'Irradiance_value'])
        knn_model = open("Madurai_model.pkl","rb")
        ml_model = joblib.load(knn_model)
        count = 0
        for row in test_data.iterrows():
            model_prediction = ml_model.predict([row[1].values[1:]])
            model_prediction = [round(float(i),2) for i in model_prediction[0]]
            model_prediction = sum(model_prediction)
            count += 1
            new_df = new_df.append({'Sno':count, 'Date' : row[1].values[0], 'Irradiance_value':model_prediction}, ignore_index = True)
        new_df.to_excel("ouput.xlsx", index = 'Sno')
        return render_template("Bulk_Forecast.html", message = "UploadSuccess")
        #return render_template("predict.html", prediction = "Success")
    return render_template("Bulk_Forecast.html", message = "UploadSuccess")

@app.route('/download', methods=['GET', 'POST'])
def downloadFile():
    path = "output.xlsx"
    return send_file(path, as_attachment=True)


if __name__ == "__main__":
    app.run(debug = True)
