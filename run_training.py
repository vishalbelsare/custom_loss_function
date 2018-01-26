'''
Train a simple deep NN on the MNIST dataset.
Get to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function
import numpy as np
from functools import partial

np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
import keras.backend as K
from itertools import product
#scikit learn
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score


num_classes = 10

# =========================================================
# Custom loss function with costs
def w_categorical_crossentropy(y_true, y_pred, weights):

    nb_cl = len(weights)
    final_mask = K.zeros_like(y_pred[:, 0])
    y_pred_max = K.max(y_pred, axis=1)
    y_pred_max = K.reshape(y_pred_max, (K.shape(y_pred)[0], 1))
    y_pred_max_mat = K.cast(K.equal(y_pred, y_pred_max), K.floatx())

    for c_p, c_t in product(range(nb_cl), range(nb_cl)):
        final_mask += (weights[c_t, c_p] * y_pred_max_mat[:, c_p] * y_true[:, c_t])

    return K.categorical_crossentropy(y_pred, y_true) * final_mask

# Define weight map for miss-classification
# TODO: because it is INDEPENDENT of the loss, thus it make more harm than good
# TODO: create weight array according to patch gt mask
w_array = np.ones((num_classes, num_classes))
w_array[1, 7] = 1.2
w_array[7, 1] = 1.2
w_array[0, 0] = 0
w_array[1, 1] = 0
w_array[2, 2] = 0
w_array[3, 3] = 0
w_array[4, 4] = 0
w_array[5, 5] = 0
w_array[6, 6] = 0
w_array[7, 7] = 0
w_array[8, 8] = 0
w_array[9, 9] = 0

ncce = partial(w_categorical_crossentropy, weights=w_array)
ncce.__name__ ='w_categorical_crossentropy'

# =========================================================

batch_size = 128
nb_classes = 10
nb_epoch = 20

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# Try halve the dataset

X_train = X_train.reshape(60000, 784)  # WHY reshape???
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices (One Hot Encoding)
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

"""
# Halve the data
X_train = X_train[0:int(len(X_train)/2)]
Y_train = Y_train[0:int(len(Y_train)/2)]
"""
# Define model
model = Sequential()
model.add(Dense(512, input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Activation('softmax'))

rms = RMSprop()
model.compile(loss=ncce, optimizer=rms, metrics=['accuracy'])
# model.compile(loss='categorical_crossentropy', optimizer=rms, metrics=['accuracy'])
# evaluate now only returns loss and not accuracy
# but passing in metrics=['accuracy'] during compile works fine
model.fit(X_train, Y_train,
          batch_size=batch_size,
          nb_epoch=nb_epoch,
          verbose=2,
          validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test,
                       verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])

# Predictions
predictions = model.predict(X_test, batch_size=batch_size, verbose=2)
predictions = np.argmax(predictions, axis=1)  # TODO: WHY axis=1 ???
"""
threshold_confusion = 0.5
pred = np.empty((predictions.shape[0]))
for i in range(predictions.shape[0]):
    if predictions[i]>=threshold_confusion:
        pred[i]=1
    else:
        pred[i]=0
"""
confusion = confusion_matrix(y_test, predictions)  # y_test & predictions must non-categorical
print(confusion)
accuracy = 0
if float(np.sum(confusion))!=0:
    accuracy = float(confusion[0,0]+confusion[1,1])/float(np.sum(confusion))
print("Global Accuracy: " +str(accuracy))
specificity = 0
if float(confusion[0,0]+confusion[0,1])!=0:
    specificity = float(confusion[0,0])/float(confusion[0,0]+confusion[0,1])
print("Specificity: " +str(specificity))
sensitivity = 0
if float(confusion[1,1]+confusion[1,0])!=0:
    sensitivity = float(confusion[1,1])/float(confusion[1,1]+confusion[1,0])
print("Sensitivity: " +str(sensitivity))
precision = 0
if float(confusion[1,1]+confusion[0,1])!=0:
    precision = float(confusion[1,1])/float(confusion[1,1]+confusion[0,1])
print("Precision: " +str(precision))
