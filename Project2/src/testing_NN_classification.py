import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from math import floor
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score as AUC_score
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
accuracy_train = []
accuracy_test = []
epochs = 100
runs = 10
eta = 1e-3
lmbd = 0
acc_test = np.zeros((runs,epochs))
acc_train = np.zeros((runs,epochs))
clf = Classification(hidden_activation='RELU')

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
                        eta=eta,
                        lmbd=lmbd,
                        debug=True)
    nn.SGD(track=['accuracy','AUC'],test_data=(X_test_scaled,Y_test),train_data=(X_train_scaled,Y_train),one_hot_encoding=True)
    AUC.append(nn.area_under_curve_test[-1])
    accuracy_test.append(nn.accuracy_each_epoch_test[-1])
    accuracy_train.append(nn.accuracy_each_epoch_train[-1])
    acc_test[i,:] = nn.accuracy_each_epoch_test
    acc_train[i,:] = nn.accuracy_each_epoch_train

AUC_mean = np.mean(AUC)
accuracy_mean = np.mean(accuracy_test)
print('AUC mean = ',AUC_mean, ' accuracy mean = ',accuracy_mean)

fig,ax = plt.subplots()
for i in range(runs):
    ax.plot(acc_test[i,:],color='crimson',label='test, mean = {:.2f}'.format(accuracy_mean))
    ax.plot(acc_train[i,:],color='navy',label='train, mean = {:.2f}'.format(np.mean(acc_train)))
    if i == 0:
        ax.legend(loc=4)
ax.set_ylabel('Accuracy')
ax.set_xlabel('Epochs')
ax.set_ylim(0.7,0.82)
fig.tight_layout()

saving = True
if saving:
    filename = '../figs/cc_testing_eta_{:.3e}_lmbd_{:.3e}_epochs{}.pdf'.format(eta,lmbd,epochs)
    print('saving figure to '+filename)
    fig.savefig(filename,bbox_inches='tight')
plt.show()