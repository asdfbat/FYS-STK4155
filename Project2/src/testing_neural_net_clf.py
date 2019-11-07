import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from math import floor
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE, roc_auc_score as AUC_score
from sklearn import preprocessing

from classification_problem import Classification
from neural_network import NeuralNetwork

mpl.rcdefaults()
plt.style.use('seaborn-darkgrid')
mpl.rcParams['figure.figsize'] = [10.0, 4.0]
mpl.rcParams['figure.dpi'] = 80
mpl.rcParams['savefig.dpi'] = 100
mpl.rcParams['font.size'] = 18

def to_one_hot(category_array):
    ca = category_array # 1D array with values of the categories
    nr_categories = np.max(ca)+1
    nr_points = len(ca)
    one_hot = np.zeros((nr_points,nr_categories),dtype=int)
    one_hot[range(nr_points),ca] = 1
    return one_hot


data_pd = pd.read_pickle("../data/credit_card_cleaned.pickle")
data = data_pd.to_numpy()
output = data[:,-1]
input_data = data[:,:-1]
output_one_hot = to_one_hot(output)

hidden_neuron_list = [16,16]
AUC = []
accuracy = []
epochs = 100
runs = 1
acc_test = np.zeros((runs,epochs))
acc_train = np.zeros((runs,epochs))
clf = Classification(hidden_activation='sigmoid')

for i in tqdm(range(runs)):
    X_train, X_test, Y_train, Y_test = train_test_split(input_data, output_one_hot,test_size=0.3)
    Scaler = preprocessing.StandardScaler()
    X_train_scaled = Scaler.fit_transform(X_train)
    X_test_scaled = Scaler.transform(X_test)
    nn = NeuralNetwork( X_train_scaled,
                        Y_train,
                        problem = clf,
                        n_hidden_neurons_list=hidden_neuron_list,
                        n_output_neurons=2,
                        epochs=epochs,
                        batch_size=256,
                        eta=1e-3,
                        lmbd=1.67,
                        debug=True)
    nn.SGD(track=['accuracy','AUC'],test_data=(X_test_scaled,Y_test),train_data=(X_train_scaled,Y_train),one_hot_encoding=True)
    AUC.append(nn.area_under_curve_test[-1])
    accuracy.append(nn.accuracy_each_epoch_test[-1])
    acc_test[i,:] = nn.accuracy_each_epoch_test
    acc_train[i,:] = nn.accuracy_each_epoch_train

AUC_mean = np.mean(AUC)
accuracy_mean = np.mean(accuracy)
print('AUC mean = ',AUC_mean, ' accuracy mean = ',accuracy_mean)
