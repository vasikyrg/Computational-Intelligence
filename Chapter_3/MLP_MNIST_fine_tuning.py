import matplotlib.pyplot as plt
import numpy
import numpy as np
import tensorflow as tf
from keras.datasets import mnist
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.utils import np_utils
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import KFold
from keras.initializers import HeNormal
from keras.callbacks import EarlyStopping
from keras import backend as K
from sklearn.metrics import confusion_matrix
import time


n_h1 = [64, 128]
n_h2 = [256, 512]
l2_val = [l2(0.1), l2(0.001), l2(0.000001)]
a_val = [0.1, 0.001, 0.000001]
lr_val = [0.1, 0.01, 0.001]
times = []  # Φόρτωση χρόνου
k = 5  # 5-fold cross val





def precision_m(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall_m(y_true, y_pred):
    """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))



(Trnx, Trny), (Tstx, Tsty) = mnist.load_data()


Trny = np_utils.to_categorical(Trny, 10)
Tsty = np_utils.to_categorical(Tsty, 10)


Trnx = Trnx / 255.0  # [0,1]
Tstx = Tstx / 255.0  # [0,1]


Trnx, valx = tf.split(Trnx, [int(48000), int(12000)], 0)
Trny, valy = tf.split(Trny, [int(48000), int(12000)], 0)


x = tf.concat([Trnx, valx], 0).numpy()
y = tf.concat([Trny, valy], 0).numpy()


grid = []
f_measures_f = []
index = 0
start = time.time()
train_split = 0.8
val_split = 0.2
fold = 0
for nh1 in range(0, len(n_h1)):
    for nh2 in range(0, len(n_h2)):
        for a in range(0, len(a_val)):
            for lr in range(0, len(lr_val)):
                print(f'At this moment n_h1=' + str(n_h1[nh1]) + ' n_h2=' + str(n_h2[nh2]) + ' a=' + str(a_val[a]) +
                      ' and lr=' + str(lr_val[lr]))
                # Grid Search
                f_measures = []
                kf = KFold(5, shuffle=True, random_state=42)  # Use for KFold classification with random_state = 42
                fold = 0
                for train, test in kf.split(x):
                    fold += 1
                    print(f'Our fold is: '+str(fold)+'/5 for n_h1=' + str(n_h1[nh1]) + ' n_h2=' + str(n_h2[nh2]) +
                          ' a='+str(a_val[a])+' and lr='+str(lr_val[lr]))
                    
                    x_train = x[train]
                    y_train = y[train]
                    x_val = x[test]
                    y_val = y[test]

                    
                    model = Sequential()
                    model.add(Flatten(input_shape=(28, 28)))
                    model.add(Dense(n_h1[nh1], activation='relu', kernel_regularizer=l2_val[a],
                                    kernel_initializer=HeNormal()))
                    model.add(Dense(n_h2[nh2], activation='relu', kernel_regularizer=l2_val[a],
                                    kernel_initializer=HeNormal()))
                    model.add(Dense(10, activation='softmax', kernel_regularizer=l2_val[a],
                                    kernel_initializer=HeNormal()))
                    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=lr_val[lr]),
                                  loss=tf.keras.losses.CategoricalCrossentropy(),
                                  metrics=[f1_m])
                    history = model.fit(x_train, y_train, batch_size=256, epochs=1000, validation_data=(x_val, y_val),
                                        callbacks=[EarlyStopping(monitor='val_f1_m', patience=200, mode="max")])
                    f_measures.append(max(history.history['val_f1_m']))

                f_measures_f.append(sum(f_measures) / len(f_measures))
                print(f'F_measures is = '+str(f_measures_f[-1]))
                grid.append((n_h1[nh1], n_h2[nh2], a_val[a], lr_val[lr]))
times.append((time.time()-start))
print(f'It lasted '+str(times[0])+' secs')
# Εύρεση του μεγίστου f_measure
max_val_of_f_measure = max(f_measures_f)
for f_value in range(0, len(f_measures_f)):
    if f_measures_f[f_value] == max_val_of_f_measure:
        print(f'Max f_measure is '+str(f_measures_f[f_value]) +
              ' with: n_h1='+str(grid[index][0])+' n_h2='+str(grid[index][1]) +
              ' a='+str(grid[index][2])+' lr='+str(grid[index][3])+'')
    index += 1

model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(128, activation='relu', kernel_regularizer=l2(1e-6),
                kernel_initializer=HeNormal()))
model.add(Dense(512, activation='relu', kernel_regularizer=l2(1e-6),
                kernel_initializer=HeNormal()))
model.add(Dense(10, activation='softmax', kernel_regularizer=l2(1e-6),
                kernel_initializer=HeNormal()))
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=[f1_m, precision_m, recall_m, tf.keras.metrics.CategoricalAccuracy()])
history = model.fit(Trnx, Trny, batch_size=256, epochs=1000, validation_data=(valx, valy),
                    callbacks=[EarlyStopping(monitor='val_f1_m', patience=200, mode="max")])


trn_acc = [history.history['categorical_accuracy'][i] * 100
           for i in range(len(history.history['categorical_accuracy']))]
val_acc = [history.history['val_categorical_accuracy'][i] * 100
           for i in range(len(history.history['val_categorical_accuracy']))]

trn_loss = history.history['loss']
val_loss = history.history['val_loss']

trn_f1_m = history.history['f1_m']
val_f1_m = history.history['val_f1_m']

plt.plot(trn_f1_m)
plt.plot(val_f1_m)

plt.title('F1 measure for h_h1=128, n_h2=512, a=1e-06, lr=0.001')
plt.xlabel('Epochs')
plt.ylabel('F1 Measure')
plt.legend(['Training F1 measure', 'Validation F1 measure'], loc='lower right')
plt.savefig('F1_measure_Optimal.png')
plt.close()

plt.plot(trn_loss)
plt.plot(val_loss)

plt.title('Losses for h_h1=128, nh2=512, a=1e-06, lr=0.001')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Training Losses', 'Validation Losses'], loc='lower right')
plt.savefig('Losses_Optimal.png')
plt.close()

plt.plot(trn_acc)
plt.plot(val_acc)

plt.title('Accuracy for h_h1=128, nh2=512, a=1e-06, lr=0.001')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Training Accuracy', 'Validation Accuracy'], loc='lower right')
plt.savefig('Accuracy_Optimal.png')
plt.close()

predict = model.predict(Tstx)
con_matrix = confusion_matrix(Tsty.argmax(axis=1), predict.argmax(axis=1))
print(con_matrix)
