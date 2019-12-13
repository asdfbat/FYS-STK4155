import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import seaborn as sns
from keras.utils import np_utils

from tqdm import trange
import os
import sys
import re
sys.path.append(os.path.realpath('.'))
from pprint import pprint

import inquirer


plt.style.use('seaborn-darkgrid')
mpl.rcParams['figure.figsize'] = [10.0, 4.0]
mpl.rcParams['figure.dpi'] = 80
mpl.rcParams['savefig.dpi'] = 100
mpl.rcParams['font.size'] = 18

X_val = np.load("../data/X_val.npy").reshape((10000,28,28))
Y_val = np.load("../data/Y_val.npy")

param_names = [
    "0. T-shirt/top",
    "1. Trouser",
    "2. Pullover",
    "3. Dress",
    "4. Coat",
    "5. Sandal",
    "6. Shirt",
    "7. Sneaker",
    "8. Bag",
    "9. Ankle boot"]

with open("Jonas_guesses.dat", "a+") as outfile:
    plt.ion()
    plt.show()
    nr_corrects = 0
    for i in range(10000):
        idx = np.random.randint(0, 10000)
        img = X_val[idx]
        plt.imshow(img)
        plt.axis("off")
        plt.draw()
        plt.pause(0.1)
            
        questions = [
            inquirer.List("item",
                        message="What item is this?",
                        choices=param_names,
                    ),
        ]

        answers = inquirer.prompt(questions)["item"]
        guessed_item = int(answers[0])
        correct_item = Y_val[idx]
        print(guessed_item, correct_item)
        if guessed_item == correct_item:
            nr_corrects += 1
            print(f"\033[92m Hurrah! You correctly guessed {answers}. \033[0m")
            print(f"\033[92m Your accuracy is now {100*nr_corrects/(i+1):.1f}% \033[0m")
        else:
            print(f"\033[91m Boooh! You guessed {answers}, but the correct item was {param_names[correct_item]}. \033[91m")
            print(f"\033[91m Your accuracy is now {100*nr_corrects/(i+1):.1f}% \033[0m")
        
        #pprint(answers)
        #pprint(guessed_item)
        
        outfile.write(str(idx) + " " + str(Y_val[idx]) + " " + str(guessed_item) + "\n")