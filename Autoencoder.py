### Import packages ###

# Keras
import keras as ks
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Conv1D, Dense, Flatten
from keras.layers import MaxPooling1D, UpSampling1D, Dropout
from keras.layers import Reshape, LeakyReLU, Lambda

# Tensorboard
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
import time

class Autoencoder:

    def __init__(self, dropout = 0.1, leak = 0.1, input_dim = 512, eps_std = 1.0):
        self.dropout_rate = dropout
        self.leak = leak
        self.input_dim = input_dim
        self.latent_dim = input_dim // 8
        self.epsilon_std = eps_std

    def load_data(self, X_train, X_val, Y_train, Y_val):
        self.X_train = X_train
        self.X_val = X_val
        self.Y_train = Y_train
        self.Y_val = Y_val

    def construct_model(self):
        # Encoding
        x = Input(shape=(512,))
        x_reshape = Reshape((512,1))(x)
        d1 = self.conv_block(x_reshape)
        d2 = self.conv_block(d1)
        flat2 = Flatten()(d2)
        lr3 = self.dense_block(flat2)
        # Decoding
        z = self.ae_block(lr3)
        lr5 = self.dense_block(z)
        r6 = Reshape((32,8))(lr5)
        d7 = self.deconv_block(r6)
        d8 = self.deconv_block(d7)
        out = self.output_block(d8)
        self.model = Model(x,out)

    def compile_model(self):
        self.model.compile( loss = 'mse',
                            optimizer = 'adam',
                            metrics = ['mse'])

    def load_tensorboard(self):
        time_str = time.strftime("%Y-%m-%d %H%M")
        self.tensorboard = TensorBoard(log_dir="../logs/{}".format(time_str))

    def load_checkpointer(self):
        mse = 'val_mean_squared_error'
        self.checkpointer = ModelCheckpoint(filepath='weights.hdf5',
                                            monitor = mse,
                                            verbose = 1,
                                            save_best_only = True)

    def load_early_stopping(self, patience):
        mse = 'val_mean_squared_error'
        self.early_stopper = EarlyStopping( monitor=mse,
                                            min_delta=0,
                                            patience=patience,
                                            verbose=0,
                                            mode='min')

    def train_model(self, epochs = 100):
        callbacks = [self.checkpointer, self.tensorboard, self.early_stopper]
        self.model.fit(
            self.X_train,
            self.Y_train,
            batch_size = 100,
            epochs = epochs,
            validation_data = (self.X_val, self.Y_val),
            callbacks = callbacks)

    def conv_block(self, input):
        dropout_rate = self.dropout_rate
        c = Conv1D( filters = 8,
                    activation = 'linear',
                    kernel_size = 6,
                    strides = 1,
                    padding = 'same',
                    kernel_initializer = 'glorot_normal')(input)
        lr = LeakyReLU(self.leak)(c)
        mp = MaxPooling1D(pool_size = 4, padding = 'same')(lr)
        d = Dropout(dropout_rate)(mp)
        return d

    def deconv_block(self, input):
        dropout_rate = self.dropout_rate
        c = Conv1D( filters = 8,
                    activation = 'linear',
                    kernel_size = 6,
                    strides = 1,
                    padding = 'same',
                    kernel_initializer = 'glorot_normal')(input)
        lr = LeakyReLU(self.leak)(c)
        us = UpSampling1D(4)(lr)
        d = Dropout(dropout_rate)(us)
        return d

    def dense_block(self, input):
        h = Dense(  256,
                    activation = 'linear',
                    kernel_initializer = 'glorot_normal')(input)
        lr = LeakyReLU(self.leak)(h)
        return lr

    def ae_block(self, input):
        z = Dense(self.latent_dim, kernel_initializer = 'glorot_normal')(input)
        return z

    def output_block(self, input):
        out_c = Conv1D( filters = 1,
                        activation = 'linear',
                        kernel_size = 6,
                        strides = 1,
                        padding = 'same',
                        kernel_initializer = 'glorot_normal')(input)
        out_lr = LeakyReLU(self.leak)(out_c)
        out = Flatten()(out_lr)
        return out