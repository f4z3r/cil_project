# `cil_project`

## Implementation Details
The currently required libraries are the following:

- `numpy`
- `matplotlib`
- `tensorflow` or `theano` or `keras` (Francesco)
- Pillow for `PIL`

### Project Structure
```
- assets
  | - testing
  |   | all data required for testing
  | - training
  |   | - data      -> all training imagery
  |   | - verify    -> verification images for training
  | - validation
  |   | - data      -> all validation imagery
  |   | - verify    -> verification images for validation
  | - test
  |   | - data      -> all test imagery
  |   | - verify    -> verification images for test
- logs
  | run.log         -> log file for warning detection
- papers
  | all relevant papers for this project
- src
  | test.py         -> contains all unit tests
  | utility.py      -> contains all utility functions
  | cnn_model.py    -> the CNN model
  | run.py          -> control file used to launch the application
  | - models        -> all CNN models implemented
  |   | cnn_model.py
- README.md         -> this file
```

### Usage
```
usage: run.py [-h] [-v] (-g | -t | -a | -u | -r) {naive}

Control program to launch all actions related to this project.

positional arguments:
  {naive}           The CNN model to be used

optional arguments:
  -h, --help        show this help message and exit
  -v, --verbose     provide verbose output
  -g, --augment     augment training image set
  -t, --train       train the given CNN
  -a, --all         augment, train and run test for the given CNN
  -u, --train-test  train and run test for the given CNN
  -r, --run         run tests of a given CNN
```

Right now simply run `run.py -v -g naive` in order to augment the training set on your local machine.

### Discussions
Please check the project and the issues for discussions.
