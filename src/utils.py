# coding=utf-8

import numpy as np


def one_hot_encode(number, max_value=15):
    """One-hot encode a number in range [0, max_value]."""
    encoded = [0.0] * (max_value + 1)
    encoded[number] = 1.0
    return encoded


def float_encode(number, max_value=127):
    """Encode an integer as float in range [0, 1]."""
    return number / float(max_value)


def encode_msg(msg):
    """Encode a message as proper input for the model."""
    return np.array(one_hot_encode(int(msg[0])) + one_hot_encode(int(msg[1]), 127) + one_hot_encode(int(msg[2]), 127) + [msg[3]], dtype=np.float32)


def encoded_to_label(msg):
    """Convert input message to label message."""
    return msg[0:16], msg[16:16+128], msg[16+128:16+256], np.array(msg[16+256])


def decode_msg(msg):
    raise NotImplementedError()


class BatchGenerator:
    def __init__(self, data, input_size=10, step=1):
        self.data = data
        self.window_size = input_size + 1  # expected output included
        self.step = step
        self.song_cursor = 0
        self.msg_cursor = self.window_size - 1
        if all(song.shape[0] < self.window_size for song in self.data):
            raise ValueError("None of the songs in the data has enough messages to satisfy requirements.")

    def __iter__(self):
        return self

    def __next__(self):
        # increment cursor
        self.msg_cursor += self.step
        # msg_cursor exceeds song's end -> next song, reset msg_cursor
        while self.msg_cursor >= self.data[self.song_cursor].shape[0]:
            self.msg_cursor = self.window_size
            self.song_cursor += 1
            if self.song_cursor == self.data.shape[0]:
                self.song_cursor = 0

        # return a tuple of (input_steps-sized msg chunk, expected msg)
        current_song = self.data[self.song_cursor]
        return (list(map(encode_msg, current_song[self.msg_cursor - self.window_size: self.msg_cursor - 1])),
                encoded_to_label(encode_msg(current_song[self.msg_cursor])))


def data_generator(datapath, batch_size, input_size, step):
    data = np.load(datapath)
    gen = BatchGenerator(data, input_size=input_size, step=step)
    while True:
        X_train, y_train = zip(*[next(gen) for _ in range(batch_size)])
        y_ch, y_nt, y_vl, y_tm = zip(*y_train)
        yield np.array(X_train), list(map(np.array, (y_ch, y_nt, y_vl, y_tm)))
