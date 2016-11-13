#!/usr/bin/env python
# coding=utf-8

import argparse
import numpy as np

from utils import BatchGenerator

parser = argparse.ArgumentParser(description='Example LSTM training script.')
parser.add_argument('data', help='Path to a file with MIDI data serialized as numpy arrays.')
args = parser.parse_args()

config = dict(
    data_path=args.data,
    input_sequence_size=10,
    input_sequence_step=1,
    observation_size=19,  # 16 for channel, 1 for pitch, 1 for velocity, 1 for time
    batch_size=128
)

def data_generator():
    data = np.load(config['data_path'])
    gen = BatchGenerator(data, input_size=config['input_sequence_size'], step=config['input_sequence_step'])
    while True:
        X_train, y_train = zip(*[next(gen) for _ in range(config['batch_size'])])
        yield np.array(X_train, dtype=np.float32), np.array(y_train, dtype=np.float32)


from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation

model = Sequential()
model.add(LSTM(
    output_dim=128,
    input_shape=(config['input_sequence_size'], config['observation_size']),
    activation='sigmoid'))
model.add(Dense(config['observation_size']))
model.add(Activation('sigmoid'))

model.compile(
    loss='binary_crossentropy',
    optimizer='adadelta'
)

model.fit_generator(data_generator(), samples_per_epoch=1e5, nb_epoch=10)

