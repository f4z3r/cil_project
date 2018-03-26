# `cil_project`

## Implementation Details
The currently required libraries are the following:

- `numpy`
- `matplotlib`
- `keras` (dependent on `tensorflow` and `h5py`)
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
  | - tests
  |   | all test cases used to check performance of solution
- logs
  | run.log         -> log file for warning detection
- papers
  | all relevant papers for this project
- src
  | test.py         -> contains all unit tests
  | utility.py      -> contains all utility functions
  | run.py          -> control file used to launch the application
  | - models        -> all CNN models implemented
  |   | cnn_model.py
- README.md         -> this file
```

### Usage
```
usage: run.py [-h] [-v | -vv | -q] [-m {cnn_lr_d,dnn_class}] [-g] [-t] [-r]
              {check} ...

Control program to launch all actions related to this project.

positional arguments:
  {check}               Test utilities
    check               Test code

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose         provide verbose output
  -vv, --very_verbose   provide even more verbose output
  -q, --quiet           provide next to no output to console
  -m {cnn_lr_d,dnn_class}, --model {cnn_lr_d,dnn_class}
                        the CNN model to be used, defaults to cnn_lr_d
  -g, --augment         augment training image set
  -t, --train           train the given CNN
  -r, --run             run a trained version of a given CNN
```

Right now simply run `run.py -v -g` in order to augment the training set on your local machine.

### Discussions
Please check the project and the issues for discussions.
