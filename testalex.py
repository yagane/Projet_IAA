#partie sur l'importation d'image
import h5py
import numpy as np
import matplotlib.pyplot as plt

label_dict = {
    0: "Disturbed Galaxies",
    1: "Merging Galaxies",
    2: "Round Smooth Galaxies",
    3: "In-between Round Smooth Galaxies",
    4: "Cigar Shaped Smooth Galaxies",
    5: "Barred Spiral Galaxies",
    6: "Unbarred Tight Spiral Galaxies",
    7: "Unbarred Loose Spiral Galaxies",
    8: "Edge-on Galaxies without Bulge",
    9: "Edge-on Galaxies with Bulge"
}

i = np.random.choice(17736)

# To get the image i and its label
with h5py.File('Galaxy10_DECals.h5', 'r') as F:
    image = F['images'][i, :, :, :]
    label = F['ans'][i]

imgplot = plt.imshow(image)
plt.show()

print("This is a galaxy of type %s" % label_dict[label])
with h5py.File('Galaxy10_DECals.h5', 'r') as F:
    imgplot = plt.imshow(F['images'][2, :, :, :])
    imgplot = plt.imshow(F['images'][3, :, :, :])



#importation de tout le bordel pour créer une ia
import time
import matplotlib.pyplot as plt
import numpy as np
import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
# from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization

from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.utils import np_utils
# from keras_sequential_ascii import sequential_model_to_ascii_printout
from keras import backend as K

# Import Tensorflow with multiprocessing
import tensorflow as tf
# import multiprocessing as mp

# Loading the CIFAR-10 datasets
from keras.datasets import cifar10

# declaration variable

batch_size = 32

num_classes = 10
nb_epochs = 30

class_names = ['Disturbed Galaxies', 'Merging Galaxies', 'Round Smooth Galaxies', 'In-between Round Smooth Galaxies',
               'Cigar Shaped Smooth Galaxies', 'Barred Spiral Galaxies', 'Unbarred Tight Spiral Galaxies',
               'Unbarred Loose Spiral Galaxies', 'Edge-on Galaxies without Bulge', 'Edge-on Galaxies with Bulge']

#17736 images donc on va split 80 20 pour les train et les tests
#17736 --> 100%
# ? --> 80%

splittrain=(17736*80)/100

#17736 --> 100%
# ? --> 20%

splittest=(17736*20)/100

print("on a donc pour le train : ",int(splittrain))
print("on a donc pour le test : ",splittest)

# definition des xtrain,ytrain et xtest,ytest
i = 0
xtrain = []
ytrain = []
xtest = []
ytest = []
for i in range(int(splittrain)):
    with h5py.File('Galaxy10_DECals.h5', 'r') as F:
        xtrain.append(F['images'][i, :, :, :])
        ytrain.append(F['ans'][i])
    print(i)

i = 0
for i in range(int(splittest)):
    with h5py.File('Galaxy10_DECals.h5', 'r') as F:
        xtest.append(F['images'][i, :, :, :])
        ytest.append(F['ans'][i])
    print(i)

Xtraintrain = []
Ytraintrain = []
Xtestest = []
Ytestest = []
for i in range(10):
    Xtraintrain.append(xtrain[i])
    Ytraintrain.append(ytrain[i])
    Xtestest.append(xtest[i])
    Ytestest.append(ytest[i])

#ici on créé le modèle tu fais ce que tu veux
# définir modele
def test_model():
    # define the model

    return model


#train le modele
test = test_model()
testfit=test.fit(x=xtrain, y=ytrain, epochs=nb_epochs, verbose=1, validation_data=(xtest, ytest), shuffle=True)

#evaluate le modele
scores = test.evaluate(xtest, ytest, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


#ici tu mets matrices confusion etc tu fais ce que tu veux