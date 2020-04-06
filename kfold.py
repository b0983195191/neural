import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.models import Sequential
from keras import models
from keras import layers

#讀取檔案 #讀取正規化檔案，需自行至Excel填入column名稱
org_data = pd.read_csv('norm_265590185.csv', encoding='utf-8')
#print(org_data.head(5))
df_org_data = pd.DataFrame(data=org_data)
print(df_org_data.head(5))
print('---------------------確認資料是否讀取進來----------------------') #ok

#定義欄位為data及targets
Brain_y = df_org_data['Brain perfusion']
#print(Brain_y)
Brain_x = df_org_data[['SBP','DBP','Brain tissue oxygen','ICP','CPP']]
print(Brain_x)
print('---------------------確認欄位是否正確----------------------') #ok

#資料分割：訓練及測試
X_train,X_test,Y_train,Y_test = train_test_split(Brain_x,Brain_y,test_size=0.3)
#print(y_test)
print(X_train.shape)
print(Y_test.shape)
print('---------------------確認資料分割是否完成----------------------') #ok

def build_model():
    model = Sequential()
    model.add(layers.Dense(32, input_shape=(X_train.shape[1],), activation="relu"))
    model.add(layers.Dense(16, activation="relu"))
    model.add(layers.Dense(1))
    # 編譯模型
    model.compile(loss="mse", optimizer="sgd",metrics=[tf.keras.metrics.MeanAbsoluteError()])
    return model

k = 4
nb_val_samples = len(X_train) // k
nb_epochs = 500
mse_scores = []
mae_scores = []
all_mae_histories = []
for i in range(k):
    print("Processing Fold #" + str(i))
    # 取出驗證資料集
    X_val = X_train[ i *nb_val_samples: ( i +1 ) *nb_val_samples]
    Y_val = Y_train[ i *nb_val_samples: ( i +1 ) *nb_val_samples]
    # 結合出訓練資料集
    X_train_p = np.concatenate(
        [X_train[: i *nb_val_samples],
         X_train[( i +1 ) *nb_val_samples:]], axis=0)
    Y_train_p = np.concatenate(
        [Y_train[: i *nb_val_samples],
         Y_train[( i +1 ) *nb_val_samples:]], axis=0)
    model = build_model()
    # 訓練模型
    history = model.fit(X_train_p, Y_train_p, epochs=nb_epochs,validation_data=(X_val,Y_val),batch_size=16, verbose=0)
    # 評估模型
    mse, mae = model.evaluate(X_val, Y_val)
    mse_scores.append(mse)
    mae_scores.append(mae) #all_scores[]

    mae_history = history.history['val_mean_absolute_error'] #作圖
    all_mae_histories.append(mae_history)

print("MSE_val: ", np.mean(mse_scores))
print("MAE_val: ", np.mean(mae_scores))

average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(nb_epochs)]
plt.plot(range(1, len(average_mae_history)+1), all_mae_histories)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()