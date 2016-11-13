"""
MIDI conversion utilities.

Current format:
- dataset is a collection of songs (np.array)
- a song is a collection of MIDI note_on messages (np.array)
- a message (np.array of float32) consists of several fields:
    a) channel (single field, int value in range 0-15)
    b) pitch (single field, int value in range 0-127)
    c) velocity (single field, int value in range 0-127)
    d) time in seconds since last message (single field, any positive float value)

So, currently a message is a np.array of shape (4, 1).
"""

__author__ = "Piotr Żelasko"
__email__ = "pzelasko@agh.edu.pl"


import numpy as np
import mido


def convert_file(path):
    """Convert a single MIDI file found in :path: to a numpy array."""
    try:
        with mido.MidiFile(path) as midi_in:
            note_messages = filter(lambda msg: msg.type in 'note_on', midi_in)
            return np.array([(msg.channel, msg.note, msg.velocity, msg.time) for msg in note_messages],
                    dtype=np.float32)
    except KeyboardInterrupt:
        raise
    except:
        # TODO: this is too broad,
        # might be worth investigating in detail what kind of errors come up
        from sys import stderr
        print("Error while reading {}".format(path), file=stderr)
        return None


def convert_batch(dirpath):
    """
    Convert all MIDI files found recursively in :dirpath: to numpy arrays.
    Expects the files to have a '.mid' extension.
    Returns an iterable of tuples (filename, array)
    """
    import os
    return filter(
        lambda result: result is not None,
        (
            convert_file(os.path.join(root, f))
            for root, dirs, files in os.walk(dirpath)
            for f in files
            if f.lower().endswith('.mid')
        )
    )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Tool for MIDI conversion to numpy arrays.')
    parser.add_argument('path', help='Path to a MIDI file or directory with MIDI files.')
    parser.add_argument('output', help='Name of the file where the serialized output will be saved.')
    args = parser.parse_args()

    from os.path import isfile, isdir
    if isfile(args.path):
        data = [convert_file(args.path)]
        if data[0] is None:
            from sys import exit, stderr
            print("File {} is corrupt.".format(args.path), file=stderr)
            exit(1)
    elif isdir(args.path):
        data = list(convert_batch(args.path))
    else:
        raise ValueError('{} does not exist.'.format(args.path))

    with open(args.output, 'wb') as f_out:
        np.save(f_out, np.array(data))
