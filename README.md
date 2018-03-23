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
usage: run.py [-h] [-m {naive}] [-g] [-t] [-r] [-c] [-v | -vv]

Control program to launch all actions related to this project.

optional arguments:
  -h, --help            show this help message and exit
  -m {naive}, --model {naive}
                        The CNN model to be used, defaults to naive
  -g, --augment         augment training image set
  -t, --train           train the given CNN
  -r, --run             run a trained version of a given CNN
  -c, --code_check      run code tests, can be run only with unittest
                        additional optional arguments
  -v, --verbose         provide verbose output
  -vv, --very_verbose   provide even more verbose output
```

Right now simply run `run.py -v -g` in order to augment the training set on your local machine.

### Discussions
Please check the project and the issues for discussions.
