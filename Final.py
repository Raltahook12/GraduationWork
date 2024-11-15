import keras.metrics
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

def norm(data):
    data = pd.DataFrame(data)
    for col in data:
        data[col] = (data[col] - data[col].min())/(data[col].max() - data[col].min())
    return data.to_numpy()

def norm_pred(data_pred,all_data):
    data = pd.DataFrame(data_pred)
    all_data = pd.DataFrame(all_data)
    for col in data:
        data[col] = (data[col] - all_data[col].min()) / (all_data[col].max() - all_data[col].min())
    return data.to_numpy()
def build_model(neurons):
    model = MyModel(neurons)
    model.compile(loss = "mae",
                  metrics = [],
                  optimizer=tf.keras.optimizers.Adam(0.1))
    return model

def denorm(data,X,y,iterator):
    X = pd.DataFrame(X)
    y = pd.DataFrame(y)
    data = pd.DataFrame(data)
    for col in X:
        X[col] = X[col] * (data[col].max() - data[col].min()) + data[col].min()
    y = y * (data[data.columns[iterator]].max() - data[data.columns[iterator]].min()) + data[data.columns[iterator]].min()
    return X.to_numpy(),y.to_numpy()

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt((K.square(y_pred - y_true))/K.int_shape(y_true)[0])

def load_data(file_path,iterator,pred = None,study_data = 0):
    if pred == None:
        data = np.loadtxt(file_path, delimiter='\t')
        norm_data = norm(data)
        X = norm_data[:, :9]
        y = norm_data[:, iterator]
        return X, y, data,norm_data
    else:
        data = np.loadtxt(file_path, delimiter='\t')
        norm_data = norm_pred(data,study_data)
        X = norm_data[:, :9]
        y = norm_data[:, iterator]
        return X, y
def swish(x):
    return K.sigmoid(0.6*x)

class MyModel(keras.Model):
    def __init__(self,neurons = 2,**kwargs):
        super().__init__(**kwargs)
        self.dense1 = keras.layers.Dense(neurons, activation=swish,use_bias=True, bias_initializer='zeros')
        self.dense2 = keras.layers.Dense(1, activation=swish)

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        return self.dense2(x)

    def get_config(self,neurons):
        return {"neurons": neurons}

X, y, data, norm_data = load_data('Alldata', -1)
model = build_model(4)
history = model.fit(X, y, epochs=1500, batch_size=len(y), validation_split=0.0, verbose=2)
Xs = []
X_pred = X[18, :]
for j in np.arange(0, 1.0, 0.1):
    X_pred[2] = j
    Xs.append(X_pred.copy())
Xs = np.array(Xs)

Y_predicted = model.predict(Xs)
fig = plt.figure()
Xs_denorm,Y_predicted_denorm = denorm(data,Xs,Y_predicted,-1)
plt.plot(Xs_denorm[:,2],Y_predicted_denorm)
plt.ylabel('Объём пор, $см^3/г$')
plt.xlabel('Разбавление изопропанолом, моль')
fig.savefig(f"Объём пор.png")

kf = KFold(n_splits=5, shuffle=True, random_state=42)
prediction = [[],[]]
for train_index, val_index in kf.split(norm_data):
    x_train, y_train = norm_data[train_index, :-1], norm_data[train_index, -1]
    x_val, y_val = norm_data[val_index, :-1], norm_data[val_index, -1]
    model = build_model(4)
    history = model.fit(X, y,epochs=1500,batch_size = len(y),validation_split=0.0,verbose=2)
    prediction[0] = [j[0] for j in model.predict(x_val).astype(float)] + prediction[0]
    prediction[1] = [j for j in y_val] + prediction[1]

fig = plt.figure()
y_pred = prediction[0]
y_true = prediction[1]
y_pred = np.array(y_pred)
y_true = np.array(y_true)
plt.scatter(y_true, y_pred)
Rscore = keras.metrics.R2Score()
Rscore.update_state(y_true.flatten().reshape(len(y_true), 1).astype(np.float32),
                    y_pred.flatten().reshape(len(y_pred), 1).astype(np.float32))
result = Rscore.result()
plt.text(.2, 0.8, round(result.numpy(),2))
plt.xlabel('Реальные значения')
plt.ylabel('Предсказанные значения')
fig.suptitle(f"Число нейронов 4")
fig.savefig(f"model4_crosval")