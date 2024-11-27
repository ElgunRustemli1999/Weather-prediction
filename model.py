import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
import functions
import numpy as np
import warnings
from sklearn.preprocessing import MinMaxScaler
import requests
import os




warnings.filterwarnings("ignore")
def model_pred(city):
    city = city.lower()

    # Dosyaları indirme fonksiyonu
    def download_file(url, local_path):
        response = requests.get(url)
        if response.status_code == 200:
            with open(local_path, 'wb') as file:
                file.write(response.content)
        else:
            raise Exception(f"Dosya indirilemedi: {url} (Status Code: {response.status_code})")

    # GitHub'dan dosya URL'leri
    model_url = f'https://raw.githubusercontent.com/ElgunRustemli1999/weather-prediction/main/models/{city}_data_model.keras'
    data_url = f'https://raw.githubusercontent.com/ElgunRustemli1999/weather-prediction/main/data/{city}_data.csv'

    # Geçici dosya yolları
    model_path = '/tmp/city_data_model.keras'
    data_path = '/tmp/city_data.csv'

    # Dosyaları indir
    download_file(model_url, model_path)
    download_file(data_url, data_path)

    # Modeli yükle
    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        raise FileNotFoundError("Model dosyası bulunamadı.")

    # Veriyi yükle ve işle
    if os.path.exists(data_path):
        data = functions.load_and_preprocess(data_path)
    else:
        raise FileNotFoundError("Veri dosyası bulunamadı.")

    # Veri hazırlığı
    feature_columns = [col for col in data.columns if col != "tavg"]
    target_column = 'tavg'
    true_value = data['tavg']

    # Veriyi ölçekle
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    scaler_x.fit(data[feature_columns])
    scaler_y.fit(data[[target_column]])
    df_x = scaler_x.transform(data[feature_columns])
    df_y = scaler_y.transform(data[[target_column]])

    df_x = df_x[-40:]
    df_y = df_y[-40:]

    # Zaman penceresi oluştur
    X_data, y_data, time_steps = functions.create_timewindow(df_x, df_y, time_steps=5)

    # Tahmin yap
    true_y_train, true_y_train_pred = functions.model_pred(model, X_data, y_data, scaler_y)

    # Sonuçları birleştir
    merged_array_col = np.hstack((true_y_train, true_y_train_pred))
    df = pd.DataFrame(merged_array_col)
    return true_y_train_pred, true_value
