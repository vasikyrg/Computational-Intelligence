import numpy as np
import tensorflow as tf
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential
from keras.utils import np_utils
from tensorflow.keras.regularizers import l1, l2
import matplotlib.pyplot as plt
from keras.initializers import RandomNormal
import time

# --> Διαχείρηση Δεδομένων
# Κατεβαίνει το dataset της εκφώνησης, εξασφαλίζεται η τυχαιότητα
(Trnx, Trny), (Tstx, Tsty) = mnist.load_data()

# Μετατροπή συστοιχιών κλάσης 1 διαστάσεων σε πίνακες κλάσης 10 διαστάσεων για την είσοδο
Trny = np_utils.to_categorical(Trny, 10)
Tsty = np_utils.to_categorical(Tsty, 10)

# Μετατροπή συστοιχιών κλάσης 1 διαστάσεων σε πίνακες κλάσης 10 διαστάσεων για την έξοδο
Trnx = Trnx / 255.0  # Γενικά παίρνει τιμές από 1 έως 255 όμως εμείς θέλουμε από [0,1]
Tstx = Tstx / 255.0  # Γενικά παίρνει τιμές από 1 έως 255 όμως εμείς θέλουμε από [0,1]

# Κάνουμε διαχωρισμό με σκοπό να έχουμε το 20% για επικύρωση
Trnx, valx = tf.split(Trnx, [int(48000), int(12000)], 0)
Trny, valy = tf.split(Trny, [int(48000), int(12000)], 0)

# --> Δημιουργία μοντέλων
batching = [48000, 256, 1]  # [48000, 256, 1]
rho = [0.01, 0.99]
regular = [l2(0.1), l2(0.01), l2(0.001)]
regular_l1 = [l1(0.01)]
r_c = [0.1, 0.01, 0.001, 0.01]
r_c_n = [2, 2, 2, 1]
times = []
# # --> Δημιουργία default μοντέλου
# for b1 in range(0, len(batching)):
#     start = time.time()
#     print("At this moment batch size is ", batching[b1], "")
#     model1 = Sequential()
#     model1.add(Flatten(input_shape=(28, 28)))
#     model1.add(Dense(128, activation='relu'))
#     model1.add(Dense(256, activation='relu'))
#     model1.add(Dense(10, activation='softmax'))
#     model1.compile(loss=tf.keras.losses.CategoricalCrossentropy(), metrics=[tf.keras.metrics.CategoricalAccuracy()])
#     history1 = model1.fit(Trnx, Trny, batch_size=batching[b1], epochs=100, validation_data=(valx, valy))
#
#     # Εκτύπωση Διαγραμμάτων και Αποθήκευση τους
#     trn_acc = [history1.history['categorical_accuracy'][i] * 100 for i in range(100)]
#     val_acc = [history1.history['val_categorical_accuracy'][i] * 100 for i in range(100)]
#     trn_los = history1.history['loss']
#     val_los = history1.history['val_loss']
#
#     plt.plot(trn_acc)
#     plt.plot(val_acc)
#
#     plt.title(f'Accuracy with batch =' + str(batching[b1]))
#     plt.xlabel('Epochs')
#     plt.ylabel('Accuracy')
#     plt.legend(['Training Accuracy', 'Validation Accuracy'], loc='lower right')
#     plt.savefig(
#         'Accuracy with batch =' + str(batching[b1]) + '.png')
#     plt.close()
#
#     plt.plot(trn_los)
#     plt.plot(val_los)
#
#     plt.title(f'Losses with batch =' + str(batching[b1]))
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend(['Training Losses', 'Validation Accuracy'], loc='upper right')
#     plt.savefig(
#         'Losses with batch =' + str(batching[b1]) + '.png')
#     plt.close()
#
#     times.append((time.time()-start))
#     print("Time for batch = " + str(batching[b1]) + " is "+str(times[b1])+" sec")
#
# # Δημιουργία μοντέλου RMSprop με βάση τις προδιαγρφές που λέει η εκφώνηση
#
# for r in range(0, len(rho)):
#     print("At this moment rho is ", rho[r], "")
#     model1 = Sequential()
#     model1.add(Flatten(input_shape=(28, 28)))
#     model1.add(Dense(128, activation='relu'))
#     model1.add(Dense(256, activation='relu'))
#     model1.add(Dense(10, activation='softmax'))
#     model1.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=rho[r]),
#                    loss=tf.keras.losses.CategoricalCrossentropy(),
#                    metrics=[tf.keras.metrics.CategoricalAccuracy()])
#     history1 = model1.fit(Trnx, Trny, batch_size=256, epochs=100, validation_data=(valx, valy))
#     # Εκτύπωση Διαγραμμάτων και Αποθήκευση τους
#     trn_acc = [history1.history['categorical_accuracy'][i] * 100 for i in range(100)]
#     val_acc = [history1.history['val_categorical_accuracy'][i] * 100 for i in range(100)]
#     trn_los = history1.history['loss']
#     val_los = history1.history['val_loss']
#
#     plt.plot(trn_acc)
#     plt.plot(val_acc)
#
#     plt.title(f'Accuracy with rho= ' + str(rho[r]))
#     plt.xlabel('Epochs')
#     plt.ylabel('Accuracy')
#     plt.legend(['Training Accuracy', 'Validation Accuracy'], loc='lower right')
#     plt.savefig('Accuracy with  rho= ' + str(rho[r])+'.png')
#     plt.close()
#
#     plt.plot(trn_los)
#     plt.plot(val_los)
#
#     plt.title(f'Losses with rho= ' + str(rho[r]))
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend(['Training Losses', 'Validation Losses'], loc='upper right')
#     plt.savefig('Losses with rho= ' + str(rho[r]) + '.png')
#     plt.close()
#
# # Δημιουργία μοντέλου SGD με βάση τις προδιαγρφές lr =0.01 με μ.ο 10
#
#     print("At this moment regularization choice is lr=0.01 with W = 10")
#     model2 = Sequential()
#     model2.add(Flatten(input_shape=(28, 28)))
#     model2.add(Dense(128, activation='relu', kernel_initializer=RandomNormal(mean=10.0/255.0)))
#     model2.add(Dense(256, activation='relu', kernel_initializer=RandomNormal(mean=10.0/255.0)))
#     model2.add(Dense(10, activation='softmax', kernel_initializer=RandomNormal(mean=10.0/255.0)))
#     model2.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
#                    loss=tf.keras.losses.CategoricalCrossentropy(),
#                    metrics=[tf.keras.metrics.CategoricalAccuracy()])
#     history2 = model2.fit(Trnx, Trny, batch_size=256, epochs=100, validation_data=(valx, valy))
#
#     # Εκτύπωση Διαγραμμάτων και Αποθήκευση τους
#     trn_acc = [history2.history['categorical_accuracy'][i] * 100 for i in range(100)]
#     val_acc = [history2.history['val_categorical_accuracy'][i] * 100 for i in range(100)]
#     trn_los = history2.history['loss']
#     val_los = history2.history['val_loss']
#
#     plt.plot(trn_acc)
#     plt.plot(val_acc)
#
#     plt.title(f'(SGD) Accuracy with lr=0.01 with W = 10')
#     plt.xlabel('Epochs')
#     plt.ylabel('Accuracy')
#     plt.legend(['Training Accuracy', 'Validation Accuracy'], loc='lower right')
#     plt.savefig('(SGD) Accuracy with lr=0.01 with W = 10 .png')
#     plt.close()
#
#     plt.plot(trn_los)
#     plt.plot(val_los)
#
#     plt.title(f'(SGD) Losses with lr=0.01 with W = 10')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend(['Training Losses', 'Validation Losses'], loc='upper right')
#     plt.savefig('(SGD) Losses with lr=0.01 with W = 10 .png')
#     plt.close()
#
# # Δημιουργία μοντέλου SGD με βάση τις προδιαγρφές lr =0.01 με μ.ο 10 και l2-norm
#     for reg2 in range(0, len(regular)):
#         print("At this moment regularization choice is l", r_c_n[reg2], " with a = ", r_c[reg2], "")
#         model2 = Sequential()
#         model2.add(Flatten(input_shape=(28, 28)))
#         model2.add(Dense(128, activation='relu', kernel_initializer=RandomNormal(mean=10.0/255.0),
#                          kernel_regularizer=regular[reg2]))
#         model2.add(Dense(256, activation='relu', kernel_initializer=RandomNormal(mean=10.0/255.0),
#                          kernel_regularizer=regular[reg2]))
#         model2.add(Dense(10, activation='softmax', kernel_initializer=RandomNormal(mean=10.0/255.0),
#                          kernel_regularizer=regular[reg2]))
#         model2.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
#                        loss=tf.keras.losses.CategoricalCrossentropy(),
#                        metrics=[tf.keras.metrics.CategoricalAccuracy()])
#         history2 = model2.fit(Trnx, Trny, batch_size=256, epochs=100, validation_data=(valx, valy))
#
#         # Εκτύπωση Διαγραμμάτων και Αποθήκευση τους
#         trn_acc = [history2.history['categorical_accuracy'][i] * 100 for i in range(100)]
#         val_acc = [history2.history['val_categorical_accuracy'][i] * 100 for i in range(100)]
#         trn_los = history2.history['loss']
#         val_los = history2.history['val_loss']
#
#         plt.plot(trn_acc)
#         plt.plot(val_acc)
#
#         plt.title(f'(SGD) Accuracy with regularization= l' + str(r_c_n[reg2]) + ' with a=' + str(r_c[reg2]))
#         plt.xlabel('Epochs')
#         plt.ylabel('Accuracy')
#         plt.legend(['Training Accuracy', 'Validation Accuracy'], loc='lower right')
#         plt.savefig('(SGD) Accuracy with regularization= l' + str(r_c_n[reg2]) + ' with a=' + str(r_c[reg2]) + '.png')
#         plt.close()
#
#         plt.plot(trn_los)
#         plt.plot(val_los)
#
#         plt.title(f'(SGD) Losses with regularization= l' + str(r_c_n[reg2]) + ' with a=' + str(r_c[reg2]))
#         plt.xlabel('Epochs')
#         plt.ylabel('Loss')
#         plt.legend(['Training Losses', 'Validation Losses'], loc='upper right')
#         plt.savefig('(SGD) Losses with regularization= l' + str(r_c_n[reg2]) + ' with a=' + str(r_c[reg2]) + '.png')
#         plt.close()

# Δημιουργία μοντέλου SGD με βάση τις προδιαγρφές lr =0.01 με μ.ο 10 και l1-norm και με dropout(0.3)
print("At this moment regularization choice is l1-norm with a = 0.01")
model2 = Sequential()
model2.add(Flatten(input_shape=(28, 28)))
model2.add(Dense(128, activation='relu', kernel_regularizer=regular_l1[0]))
model2.add(Dropout(0.3))
model2.add(Dense(256, activation='relu', kernel_regularizer=regular_l1[0]))
model2.add(Dropout(0.3))
model2.add(Dense(10, activation='softmax', kernel_regularizer=regular_l1[0]))
model2.compile(optimizer=tf.keras.optimizers.SGD(), loss=tf.keras.losses.CategoricalCrossentropy(),
               metrics=[tf.keras.metrics.CategoricalAccuracy()])
history2 = model2.fit(Trnx, Trny, batch_size=256, epochs=100, validation_data=(valx, valy))

# Εκτύπωση Διαγραμμάτων και Αποθήκευση τους
trn_acc = [history2.history['categorical_accuracy'][i] * 100 for i in range(100)]
val_acc = [history2.history['val_categorical_accuracy'][i] * 100 for i in range(100)]
trn_los = history2.history['loss']
val_los = history2.history['val_loss']

plt.plot(trn_acc)
plt.plot(val_acc)

plt.title(f'(SGD) Accuracy with regularization choice is l1-norm with a = 0.01')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Training Accuracy', 'Validation Accuracy'], loc='lower right')
plt.savefig('(SGD) Accuracy with regularization choice is l1-norm with a = 0.01.png')
plt.close()

plt.plot(trn_los)
plt.plot(val_los)

plt.title(f'(SGD) Losses with regularization choice is l1-norm with a = 0.01')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Training Accuracy', 'Validation Accuracy'], loc='upper right')
plt.savefig('(SGD) Losses with regularization choice is l1-norm with a = 0.01.png')
plt.close()
