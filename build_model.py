from keras.models import Sequential
from keras.models import model_from_json
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping
import keras.backend as K
from keras.utils import plot_model


TIME_STEPS = 8  # 输入的时间步数
INPUT_SIZE = 33  # 每步有多少数据
OUTPUT_SIZE = 33  # 输出的维度
EPOCHS = 10  # 迭代次数
BATCH_SIZE = 64

def build_lstm():
    model = Sequential()
    model.add(LSTM(input_shape=(TIME_STEPS, INPUT_SIZE),
                   output_dim=64,
                   return_sequences=True, ))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))
    model.add(LSTM(output_dim=256))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(OUTPUT_SIZE))

    return model