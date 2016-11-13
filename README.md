# Music generation with MIDI and Keras

## Setup

This code depends on:
- python 3.5 (not tested on python 2)
- numpy
- keras (tested with tensorflow backend)
- mido (for MIDI parsing)

## Usage

To convert MIDI files to numpy arrays, use:

    python midi2numpy.py my_midi_directory file_with_output.dat

To train LSTM:

    python lstm.py file_with_output.dat

