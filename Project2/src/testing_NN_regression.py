import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from math import floor
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE, roc_auc_score as AUC_score
from sklearn import preprocessing

from regression_problem import Regression
from neural_network import NeuralNetwork

mpl.rcdefaults()
plt.style.use('seaborn-darkgrid')
mpl.rcParams['figure.figsize'] = [10.0, 4.0]
mpl.rcParams['figure.dpi'] = 80
mpl.rcParams['savefig.dpi'] = 100
mpl.rcParams['font.size'] = 18


parameter_names = ["Happiness", "Economy", "Family", "Health", "Freedom", "Trust", "Generosity"]
data_df = pd.read_pickle("../data/world_happiness.pickle")[parameter_names]
data = data_df.to_numpy()
output = data[:,0].reshape(-1,1)
input_data = data[:,1:]

print(output.shape,'output shape')
print(input_data.shape,'input shape')

hidden_neuron_list = [16,16]
epochs = 100
runs = 10
r2_test_runs = np.zeros((runs,epochs))
r2_train_runs = np.zeros((runs,epochs))
r2_end_test = np.zeros(epochs)
r2_end_train = np.zeros(epochs)
reg = Regression(hidden_activation='RELU')

for i in tqdm(range(runs)):
    X_train, X_test, Y_train, Y_test = train_test_split(input_data, output,test_size=0.1)
    Scaler = preprocessing.StandardScaler()
    X_train_scaled = Scaler.fit_transform(X_train)
    X_test_scaled = Scaler.transform(X_test)
    nn = NeuralNetwork( X_train_scaled,
                        Y_train,
                        problem=reg,
                        n_hidden_neurons_list=hidden_neuron_list,
                        n_output_neurons=1,
                        epochs=epochs,
                        batch_size=32,
                        eta=1e-3,
                        lmbd=0,
                        debug=True)
    
    nn.SGD(track=['r2'],test_data=(X_test_scaled,Y_test),train_data=(X_train_scaled,Y_train),one_hot_encoding=False)
    r2_test_runs[i,:] = nn.r2_test
    r2_train_runs[i,:] = nn.r2_train
    r2_end_test[i] = nn.r2_test[-1]
    r2_end_train[i] = nn.r2_train[-1]

r2_mean_test = np.mean(r2_end_test)
r2_mean_train = np.mean(r2_end_train)
print('r2 mean test = ',r2_mean_test, ' r2 mean train = ',r2_mean_train)

fig,ax = plt.subplots()
for i in range(runs):
    ax.plot(r2_test_runs[i,:],label='test')
    ax.plot(r2_train_runs[i,:],label='train')
ax.axhline(y=1)
plt.show()