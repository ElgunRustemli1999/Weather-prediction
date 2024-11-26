import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


def load_and_preprocess(data_path ):
  data_path =data_path
  data = pd.read_csv(data_path) 
  print(data.columns)
  data['date'] = pd.to_datetime(data['date'])
  for col in data.columns:
    if data[col].isnull().sum() > data.shape[0]*0.3:
      data = data.drop(columns=[col])
    else:
      data[col] = data[col].ffill().bfill()



  data.set_index('date', inplace=True)
  return data.resample('D').mean()

def split_data_by_time(df, train_ratio=0.7, val_ratio=0.15):
  n = len(df)
  train_size = int(train_ratio * n)
  val_size = int(val_ratio * n)
  test_size = n- train_size - val_size
  train_data = df[:train_size]
  val_data = df[train_size:train_size+val_size]
  test_data = df[train_size+val_size:]
  return train_data, val_data, test_data

def scale_data(df,feature_columns,target_column,nums=None):
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    scaler_x.fit(df[feature_columns])
    scaler_y.fit(df[[target_column]])
    df_x = scaler_x.transform(df[feature_columns])
    df_y = scaler_y.transform(df[[target_column]])
    if nums is None:
       nums=7
    return df_x[-nums], df_y[-nums]
    


def scale_all_data (train_data, val_data, test_data, feature_columns, target_column):
  scaler_x = MinMaxScaler()
  scaler_y = MinMaxScaler()
  scaler_x.fit(train_data[feature_columns])
  scaler_y.fit(train_data[[target_column]])
  train_x = scaler_x.transform(train_data[feature_columns])
  train_y = scaler_y.transform(train_data[[target_column]])
  val_x = scaler_x.transform(val_data[feature_columns])
  val_y = scaler_y.transform(val_data[[target_column]])
  test_x = scaler_x.transform(test_data[feature_columns])
  test_y = scaler_y.transform(test_data[[target_column]])
  return train_x, val_x, test_x, train_y.flatten(), val_y.flatten(), test_y.flatten(), scaler_x, scaler_y

def create_timewindow(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X[i:(i + time_steps)]
        Xs.append(v)
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys), time_steps
def plot_training_history(history, train_loss='loss', train_metric='accuracy', val_loss='val_loss', val_metric='val_accuracy'):

    #Loss
    plt.figure(figsize=(10, 5))
    plt.plot(history.history[train_loss], label='Training Loss')
    plt.plot(history.history[val_loss], label='Validation Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Metrics
    plt.figure(figsize=(10, 5))
    plt.plot(history.history[train_metric], label=f"Training: {train_metric}")
    plt.plot(history.history[val_metric], label=f"Validation: {val_metric}")
    plt.title(f'Training and Validation {train_metric} Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel(f'train_metric')
    plt.legend()
    plt.show()
def model_pred(model,data_x,data_y,scaler_y):
   pred = model.predict(data_x)
   true_y_train_pred = scaler_y.inverse_transform(pred)
   true_y_train = scaler_y.inverse_transform(data_y.reshape(-1, 1))
   return true_y_train_pred,true_y_train
def evaluate_predictions(model, X_train, y_train, X_val, y_val, X_test, y_test, scaler_y):

    y_train_pred_ = model.predict(X_train, verbose=0)
    y_val_pred = model.predict(X_val, verbose=0)
    y_test_pred = model.predict(X_test, verbose=0)

    true_y_train_pred = scaler_y.inverse_transform(y_train_pred_)
    true_y_val_pred = scaler_y.inverse_transform(y_val_pred)
    true_y_test_pred = scaler_y.inverse_transform(y_test_pred)

    true_y_train = scaler_y.inverse_transform(y_train.reshape(-1, 1))
    true_y_val = scaler_y.inverse_transform(y_val.reshape(-1, 1))
    true_y_test = scaler_y.inverse_transform(y_test.reshape(-1, 1))

    train_mse = mean_squared_error(true_y_train, true_y_train_pred)
    val_mse = mean_squared_error(true_y_val, true_y_val_pred)
    test_mse = mean_squared_error(true_y_test, true_y_test_pred)
    print("Train MSE:", train_mse, "Validation MSE:", val_mse, "Test MSE:", test_mse)
    return true_y_train, true_y_train_pred, true_y_val, true_y_val_pred, true_y_test, true_y_test_pred, train_mse, val_mse, test_mse



def plot_predictions_with_metrics(train_df, val_df, test_df, true_y_train, true_y_train_pred, true_y_val, true_y_val_pred, true_y_test, true_y_test_pred, train_mse, val_mse, test_mse, time_steps):
    plt.figure(dpi=200)
    plt.figure(figsize=(30, 10))
    # Train
    train_time = train_df.index[time_steps:]
    plt.plot(train_time, true_y_train, label='True Values (Train)', color='blue')
    plt.plot(train_time, true_y_train_pred, '--', label='Predicted Values (Train)', color='red')

    # Validation
    val_time = val_df.index[time_steps:]
    plt.plot(val_time, true_y_val, label='True Values (Validation)', color='green')
    plt.plot(val_time, true_y_val_pred, '--', label='Predicted Values (Validation)', color='orange')

    # Test
    test_time = test_df.index[time_steps:]
    plt.plot(test_time, true_y_test, label='True Values (Test)', color='purple')
    plt.plot(test_time, true_y_test_pred, '--', label='Predicted Values (Test)', color='pink')

    # Metrics
    plt.text(train_time.min(), true_y_train.max()*1.03, f"Train MSE: {train_mse:.4f}", fontsize=12)
    plt.text(train_time.min(), true_y_train.max()*0.98, f"Validation MSE: {val_mse:.4f}", fontsize=12)
    plt.text(train_time.min(), true_y_train.max()*0.93, f"Test MSE: {test_mse:.4f}", fontsize=12)

    plt.title('Actual and Predicted Values for All Data Sets')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()