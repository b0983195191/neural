import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras import models
from keras import layers
from keras.callbacks import EarlyStopping

# root mean squared error (rmse) for regression
def rmse(y_true, y_pred):
    from keras import backend
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

# mean squared error (mse) for regression
def mse(y_true, y_pred):
    from keras import backend
    return backend.mean(backend.square(y_pred - y_true), axis=-1)

# coefficient of determination (R^2) for regression
def r_square(y_true, y_pred):
    from keras import backend as K
    SS_res =  K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res/(SS_tot + K.epsilon()))

def r_square_loss(y_true, y_pred):
    from keras import backend as K
    SS_res =  K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - ( 1 - SS_res/(SS_tot + K.epsilon()))

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

model = Sequential()
model.add(layers.Dense(32, input_shape=(X_train.shape[1],), activation="relu"))
#model.add(layers.Dense(16, activation="relu"))
model.add(layers.Dense(1))
# 編譯模型
model.compile(loss="mse", optimizer="adam",metrics=["mean_squared_error", rmse, r_square])
# enable early stopping based on mean_squared_error
earlystopping=EarlyStopping(monitor="mean_squared_error", patience=40, verbose=1, mode='auto')
# fit model
result = model.fit(X_train, Y_train, epochs=80, batch_size=10, validation_data=(X_test, Y_test), callbacks=[earlystopping])
# get predictions
y_pred = model.predict(X_test)

import sklearn.metrics, math
print("\n")
print("Mean absolute error (MAE):      %f" % sklearn.metrics.mean_absolute_error(Y_test,y_pred))
print("Mean squared error (MSE):       %f" % sklearn.metrics.mean_squared_error(Y_test,y_pred))
print("R square (R^2):                 %f" % sklearn.metrics.r2_score(Y_test,y_pred))
#-----------------------------------------------------------------------------
# Plot learning curves including R^2 and RMSE
#-----------------------------------------------------------------------------
# plot training curve for R^2 (beware of scale, starts very low negative)
# plt.plot(result.history['val_r_square'])
# plt.plot(result.history['r_square'])
# plt.title('model R^2')
# plt.ylabel('R^2')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()

# # plot training curve for rmse
# plt.plot(result.history['rmse'])
# plt.plot(result.history['val_rmse'])
# plt.title('rmse')
# plt.ylabel('rmse')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
#
# # print the linear regression and display datapoints
# from sklearn.linear_model import LinearRegression
# regressor = LinearRegression()
# regressor.fit(Y_test.reshape(-1,1), y_pred)
# y_fit = regressor.predict(y_pred)
#
# reg_intercept = round(regressor.intercept_[0],4)
# reg_coef = round(regressor.coef_.flatten()[0],4)
# reg_label = "y = " + str(reg_intercept) + "*x +" + str(reg_coef)
#
# plt.scatter(Y_test, y_pred, color='blue', label= 'data')
# plt.plot(y_pred, y_fit, color='red', linewidth=2, label = 'Linear regression\n'+reg_label)
# plt.title('Linear Regression')
# plt.legend()
# plt.xlabel('observed')
# plt.ylabel('predicted')
# plt.show()