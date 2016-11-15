#!/usr/bin/env python
# coding=utf-8

import argparse
parser = argparse.ArgumentParser(description='Example LSTM training script.')
parser.add_argument('data', help='Path to a file with MIDI data serialized as numpy arrays.')
args = parser.parse_args()

config = dict(
    data_path=args.data,
    input_sequence_size=10,
    input_sequence_step=1,
    observation_size=19,  # 16 for channel, 1 for pitch, 1 for velocity, 1 for time
    batch_size=64
)

from keras.models import Model
from keras.layers import LSTM, Dense, Input

input_layer = Input(shape=(config['input_sequence_size'], config['observation_size']))
x = LSTM(
    output_dim=128,
    activation='sigmoid')(input_layer)
channel = Dense(16, activation='softmax', name='channel')(x)
note_and_velocity = Dense(2, activation='sigmoid', name='note_and_velocity')(x)
time = Dense(1, activation='linear', name='time')(x)
model = Model(input=input_layer, output=[channel, note_and_velocity, time])

model.compile(
    optimizer='adadelta',
    loss={'channel': 'categorical_crossentropy',
        'note_and_velocity': 'binary_crossentropy',
        'time': 'mse'},
    loss_weights={'channel': 1/16, 'note_and_velocity': 1/2, 'time': 100}
)

from utils import data_generator
data_gen = data_generator(config['data_path'], config['batch_size'], config['input_sequence_size'], config['input_sequence_step'])
#from utils import BatchGenerator
#import numpy as np
#data_gen = BatchGenerator(np.load(config['data_path']), config['input_sequence_size'], config['input_sequence_step'])
#X_train, y_train = zip(*[next(data_gen) for _ in range(2 ** 16)])
#X_train = np.array(X_train, dtype=np.float32)
#y1, y2, y3 = map(np.array, zip(*y_train))
#
#model.fit(X_train, [y1, y2, y3], nb_epoch=20)
model.fit_generator(data_gen, samples_per_epoch=1e5, nb_epoch=10)

model.save('prototype.mdl')

