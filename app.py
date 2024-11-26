from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import functions
import model

# Flask uygulaması
app = Flask(__name__)




# Örnek şehir verisi (gerçek verilerle değiştirilmelidir)
citys = ['Baku','Berlin','Paris','Roma','Antalya','Ankara','İstanbul']
# Ana sayfa için rota
@app.route('/')
def index():
    return render_template("index.html")  # Frontend sayfanız

# Tahmin rotasıdwa  
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Kullanıcıdan şehir ve tarih bilgisi al
        city = request.form.get('city')  # Şehir seçimi
        start_date = request.form.get('start_date')  # Başlangıç tarihi
        end_date = request.form.get('end_date')  # Bitiş tarihi
        # Tarih aralığı oluştur
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
        
        if start_date > end_date:
            return jsonify({"error": "Bitiş tarihi başlangıç tarihinden önce olamaz"}), 400
        date_range = [(start_date + timedelta(days=i)).strftime("%Y-%m-%d")
                      for i in range((end_date - start_date).days + 1)]
        total_days = len(date_range)

        # Şehir verisini al
        if city not in citys:
            return jsonify({"error": "Geçersiz şehir"}), 400
        else:

        # Model tahmini için veri oluştur
            predictions = []
            prediction,true = model.model_pred(city)
            print(prediction)
            today = datetime.today()
            today = today.strftime("%Y-%m-%d")
            print('start date type',type(start_date))
            today = today.split('-')
            today_day = int(today[-1])
            today_month = int(today[1])
            start_day = int(start_date.day)
            start_month = int(start_date.month)
            print("start_day",start_day)
            print("s_m",start_month)
            print('t_d',today_day)
            print('t_m',today_month)
            print("today tip",type(today))
            print("today",today)
            
            if start_day>=today_day or start_month>=today_month:
                i=(start_date-datetime.now()).days
                print('days i',i)
                for i, day in enumerate(date_range,start=i):
                    print(i)
                    predictions.append({
                    "date": day,
                    "temperature": float(prediction[i][0]),  # Tahmini sıcaklık
                })
            else:
                for i, day in enumerate(date_range):
                    predictions.append({
                    "date": day,
                    "temperature": float(true[i])})

            #for day in date_range:
            
            # Örnek giriş verisi: [latitude, longitude, day_of_year]
                #day_of_year = datetime.strptime(day, "%Y-%m-%d").timetuple().tm_yday
                #print("day ",day,"i",i)
                #predictions.append({
                
                #"date": day,
                #"temperature": float(prediction[i][0]),  # Tahmini sıcaklık
                
                #})
                #i=i+1
        
        # Tahminleri JSON olarak döndür
            return jsonify(predictions)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
