from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential
from keras.datasets import boston_housing
from keras import layers
from keras.initializers import Initializer
from sklearn.cluster import KMeans

# Useful link -> https://github.com/PetraVidnerova/rbf_keras


class RBFLayer(layers.Layer):
    # output_dim: number of hidden units (number of outputs of the layer)
    # initializer: instance of initializer to initialize centers
    # betas: float, initial value for betas (beta = 1 / 2*pow(sigma,2))
    def __init__(self, output_dim, initializer, betas=1.0, **kwargs):
        self.betas = betas
        self.output_dim = output_dim
        self.initializer = initializer
        super(RBFLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.centers = self.add_weight(name='centers', shape=(self.output_dim, input_shape[1]),
                                       initializer=self.initializer, trainable=False)
        d_max = 0
        for i in range(0, self.output_dim):
            for j in range(0, self.output_dim):
                d = np.linalg.norm(self.centers[i] - self.centers[j])
                if d > d_max:
                    d_max = d
        sigma = d_max / np.sqrt(2 * self.output_dim)
        self.betas = np.ones(self.output_dim) / (2 * (sigma ** 2))
        super(RBFLayer, self).build(input_shape)

    def call(self, inputs, *args, **kwargs):
        C = tf.expand_dims(self.centers, -1)  # εισάγουμε μια διάσταση από άσσους
        H = tf.transpose(C - tf.transpose(inputs))  # Πίνακας με τις διαφορές
        return tf.exp(-self.betas * tf.math.reduce_sum(H ** 2, axis=1))

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_dim


class InitCentersKMeans(Initializer):
    def __init__(self, X, max_iter=100):
        self.X = X
        self.max_iter = max_iter
        super().__init__()

    def __call__(self, shape, dtype=None, *args):
        assert shape[1] == self.X.shape[1]
        n_centers = shape[0]
        km = KMeans(n_clusters=n_centers, max_iter=self.max_iter, verbose=0)
        km.fit(self.X)
        return km.cluster_centers_


def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def mse(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true))


def r2_score(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res/(SS_tot + K.epsilon())



(Trnx, Trny), (Tstx, Tsty) = boston_housing.load_data(test_split=0.25)


Trnx = (Trnx - np.mean(Trnx, axis=0)) / np.std(Trnx, axis=0)
Tstx = (Tstx - np.mean(Tstx, axis=0)) / np.std(Tstx, axis=0)

' Η διαφορά με πριν είναι ότι πριν είχαμε classification ενώ τώρα έχουμε regression'
' δηλαδή βγαίνει ένας αριθμός απευθείας'


Trnx, valx = tf.split(Trnx, [int(Trnx.shape[0]*0.8), Trnx.shape[0] - int(Trnx.shape[0]*0.8)], 0)
Trny, valy = tf.split(Trny, [int(Trny.shape[0]*0.8), Trny.shape[0] - int(Trny.shape[0]*0.8)], 0)


neuron_1 = 0.1*Trny.shape[0]
neuron_2 = 0.5*Trny.shape[0]
neuron_3 = 0.9*Trny.shape[0]

neurons = [neuron_1, neuron_2, neuron_3]
i = 1
types_of_neurons = ["10%", "50%", "90%"]
for n in neurons:
    if i == 1:
        print(f'First neuron type')
    if i == 2:
        print(f'Second neuron type')
    if i == 3:
        print(f'Third neuron type')
    model = Sequential()
    model.add(RBFLayer(int(n), initializer=InitCentersKMeans(Trnx), input_shape=(13, )))
    model.add(Dense(128))
    model.add(Dense(1))

    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001), loss=mse, metrics=[r2_score, rmse])
    history = model.fit(Trnx, Trny, batch_size=4, epochs=100, validation_data=(valx, valy))
    score = model.evaluate(Tstx, Tsty)
    
    
    trn_loss = [history.history['loss'][i] for i in range(100)]
    val_loss = [history.history['val_loss'][i] for i in range(100)]

    trn_r2 = history.history['r2_score']
    val_r2 = history.history['val_r2_score']

    trn_rmse = history.history['rmse']
    val_rmse = history.history['val_rmse']

    plt.plot(trn_loss)
    plt.plot(val_loss)

    plt.title('Learning Curve for '+str(types_of_neurons[i-1])+' of training dataset and lr=0.001')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['Training Loss', 'Validation Loss'], loc='upper right')
    plt.savefig('RBF_'+str(types_of_neurons[i-1])+'_loss.png')
    plt.close()

    plt.plot(trn_r2)
    plt.plot(val_r2)

    plt.title('R2 for '+str(types_of_neurons[i-1])+' of training dataset and lr=0.001')
    plt.xlabel('Epochs')
    plt.ylabel('R2')
    plt.legend(['Training R2', 'Validation R2'], loc='lower right')
    plt.savefig('RBF_'+str(types_of_neurons[i-1])+'_R2.png')
    plt.close()

    plt.plot(trn_rmse)
    plt.plot(val_rmse)

    plt.title('RMSE for ' + str(types_of_neurons[i - 1]) + ' of training dataset and lr=0.001')
    plt.xlabel('Epochs')
    plt.ylabel('RMSE')
    plt.legend(['Training RMSE', 'Validation RMSE'], loc='lower right')
    plt.savefig('RBF_' + str(types_of_neurons[i - 1]) + '_RMSE.png')
    plt.close()

    if i == 1:
        print(f'MSE for the first neuron type is '+str(score[0]))
        print(f'R2 for the first neuron type is ' + str(score[1]))
        print(f'RMSE for the first neuron type is ' + str(score[2]))
    if i == 2:
        print(f'MSE for the second neuron type is '+str(score[0]))
        print(f'R2 for the second neuron type is '+str(score[1]))
        print(f'RMSE for the second neuron type is ' + str(score[2]))
    if i == 3:
        print(f'MSE for the third neuron type is '+str(score[0]))
        print(f'R2 for the third neuron type is '+str(score[1]))
        print(f'RMSE for the third neuron type is ' + str(score[2]))

    i = i + 1
