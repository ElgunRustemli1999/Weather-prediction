import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
import functions
import numpy as np
import warnings
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings("ignore")
def model_pred(city):
    city = city.lower()
    model_path = f'C:/Users/Gold/Desktop/different_weather_app/models/{city}_data_model.keras'
    try:
        model = load_model(model_path)
        print("Model başarıyla yüklendi.")
    except Exception as e:
        print(f"Model yüklenirken hata oluştu: {e}")
    data_path = f'C:/Users/Gold/Desktop/different_weather_app/data/{city}_data.csv'

    data = functions.load_and_preprocess(data_path)

    feature_columns = [col for col in data.columns if col !="tavg "]
    target_column = 'tavg'
    true_value = data['tavg']
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    scaler_x.fit(data[feature_columns])
    scaler_y.fit(data[[target_column]])
    df_x = scaler_x.transform(data[feature_columns])
    df_y = scaler_y.transform(data[[target_column]])
    
    df_x = df_x[-40:]
    df_y = df_y[-40:]
    
    X_data, y_data, time_steps = functions.create_timewindow(df_x, df_y, time_steps = 7)

    true_y_train, true_y_train_pred = functions.model_pred(model, X_data,y_data, scaler_y)
    merged_array_col = np.hstack((true_y_train, true_y_train_pred))
    print("{:.1f}".format(true_y_train_pred[0][0]))
    df = pd.DataFrame(merged_array_col)
    return true_y_train_pred, true_value
