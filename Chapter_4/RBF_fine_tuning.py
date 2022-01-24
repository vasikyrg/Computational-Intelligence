from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.datasets import boston_housing
from keras import layers
from keras.initializers import Initializer
from sklearn.cluster import KMeans
import time
from sklearn.model_selection import KFold

# Usefull link -> https://github.com/PetraVidnerova/rbf_keras


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


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true))


def r2_score(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res/(SS_tot + K.epsilon())


# Σταθερές τιμές εκφώνησης
n_h2 = [32, 64, 128, 256]
prop = [0.2, 0.35, 0.5]
times = []  # Φόρτωση χρόνου
k = 5  # 5-fold cross val


(Trnx, Trny), (Tstx, Tsty) = boston_housing.load_data(test_split=0.25)


Trnx = (Trnx - np.mean(Trnx, axis=0)) / np.std(Trnx, axis=0)
Tstx = (Tstx - np.mean(Tstx, axis=0)) / np.std(Tstx, axis=0)



Trnx, valx = tf.split(Trnx, [int(Trnx.shape[0]*0.8), Trnx.shape[0] - int(Trnx.shape[0]*0.8)], 0)
Trny, valy = tf.split(Trny, [int(Trny.shape[0]*0.8), Trny.shape[0] - int(Trny.shape[0]*0.8)], 0)


x = tf.concat([Trnx, valx], 0).numpy()
y = tf.concat([Trny, valy], 0).numpy()


neuron_1 = 0.05*Trny.shape[0]
neuron_2 = 0.15*Trny.shape[0]
neuron_3 = 0.3*Trny.shape[0]
neuron_4 = 0.5*Trny.shape[0]
neurons = [neuron_1, neuron_2, neuron_3, neuron_4]

grid = []
index = 0
start = time.time()
rmse_f = []
types_of_neurons = ["5%", "15%", "30%", "50%"]
for p in range(0, len(prop)):
    for nh2 in range(0, len(n_h2)):
        n_i = 0
        for n in range(0, len(neurons)):
            n_i = n_i + 1
            if n_i == 1:
                print(f'First neuron type')
            if n_i == 2:
                print(f'Second neuron type')
            if n_i == 3:
                print(f'Third neuron type')
            if n_i == 4:
                print(f'Fourth neuron type')
            # Grid Search
            rmse = []
            kf = KFold(5, shuffle=True, random_state=42)  # Use for KFold classification with random_state = 42
            fold = 0
            for train, test in kf.split(x):
                fold += 1
                print(f'Our fold is: ' + str(fold) + '/5 for prop=' + str(prop[p]) + ', n_h2=' + str(n_h2[nh2]) +
                      ' and its neuron type is '+str(types_of_neurons[n_i-1])+' οf dataset.')
                
                x_train = x[train]
                y_train = y[train]
                x_val = x[test]
                y_val = y[test]
                model = Sequential()
                model.add(RBFLayer(int(neurons[n]), initializer=InitCentersKMeans(x_train), input_shape=(13, )))
                model.add(Dense(n_h2[nh2]))
                model.add(Dropout(prop[p]))
                model.add(Dense(1))

                model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001), loss=mean_squared_error,
                              metrics=[r2_score, root_mean_squared_error])
                history = model.fit(x_train, y_train, batch_size=4, epochs=100, validation_data=(valx, valy))

                rmse.append(min(history.history['root_mean_squared_error']))

            rmse_f.append(sum(rmse) / len(rmse))
            print(f'RMSE is = ' + str(rmse_f[-1]))
            grid.append((types_of_neurons[n_i-1], prop[p], n_h2[nh2]))

times.append((time.time()-start))
print(f'It lasted '+str(times[0])+' secs')
# Εύρεση του ελαχίστου loss
min_val_of_loss_f = min(rmse_f)
for rmse_value in range(0, len(rmse_f)):
    if rmse_f[rmse_value] == min_val_of_loss_f:
        print(f'Min RMSE is '+str(rmse_f[rmse_value])+' with: n_h2='+str(grid[index][2]) +
              ', probability='+str(grid[index][1])+' for neuron type '+str(grid[index][0])+'.')
    index += 1

# Optimal 

model = Sequential()
model.add(RBFLayer(int(neuron_4), initializer=InitCentersKMeans(Trnx), input_shape=(13, )))
model.add(Dense(256))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001), loss=mean_squared_error,
              metrics=[r2_score, root_mean_squared_error])
history = model.fit(Trnx, Trny, batch_size=4, epochs=100, validation_data=(valx, valy))
score = model.evaluate(Tstx, Tsty)


trn_loss = [history.history['loss'][i] for i in range(100)]
val_loss = [history.history['val_loss'][i] for i in range(100)]

trn_r2 = history.history['r2_score']
val_r2 = history.history['val_r2_score']

trn_rmse = history.history['root_mean_squared_error']
val_rmse = history.history['val_root_mean_squared_error']

plt.plot(trn_loss)
plt.plot(val_loss)

plt.title('Learning Curve for Optimal Model')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Training Loss', 'Validation Loss'], loc='upper right')
plt.savefig('RBF_optimal_loss.png')
plt.close()

plt.plot(trn_r2)
plt.plot(val_r2)

plt.title('R2 for Optimal Model')
plt.xlabel('Epochs')
plt.ylabel('R2')
plt.legend(['Training R2', 'Validation R2'], loc='lower right')
plt.savefig('RBF_optimal_R2.png')
plt.close()

plt.plot(trn_rmse)
plt.plot(val_rmse)

plt.title('RMSE for Optimal Model')
plt.xlabel('Epochs')
plt.ylabel('RMSE')
plt.legend(['Training RMSE', 'Validation RMSE'], loc='lower right')
plt.savefig('RBF_optimal_RMSE.png')
plt.close()

print(f'MSE for that type of neuron type is '+str(score[0]))
print(f'R2 for that type of neuron type is '+str(score[1]))
print(f'RMSE for that type of neuron type is ' + str(score[2]))
