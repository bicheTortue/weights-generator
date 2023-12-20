# Weights generator
This repo contains the python scripts used to train the weights and save them in a file used in my [master thesis](https://github.com/bicheTortue/MSc-thesis/releases/download/Final/thesis.pdf).

## Table of Contents

- [Usage](#usage)

- [Features](#features)

- [License](#license)

## Usage

The weights for the Airline or the [C. elegans](https://doi.org/10.1038/s41598-022-25421-w) problems are trained by running [airline.py](./airline.py) or [celegans.py](./celegans.py) with `--train --save` as arguments.

Training weights for other problems is possible. You can use the [`barbaLib.py`](./barbaLib.py) module to use the custom activation functions or save the weights and adapt the code according to your specific problem.

## Feature

Those scripts feature the possibility to choose the type of RNN, the input size, the number of time steps, the number of hidden state, the size of the output and more from a command argument.

## License

This project is licensed under the General Public License, version 3.0 or later - see the [COPYING](./COPYING) file for details.
