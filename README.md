# Road Segmentation from Aerial images

This is the projects source code of Robin Bader, Francesco Saverio Varini and Jakob Beckmann which is part of a Kaggle competition that is available under this link [this link](https://www.kaggle.com/c/cil-road-segmentation-2018/).

## Quick start

To quickly get started and train and predict the final submission obtained in the Kaggle competition executed following code in the command line assuming that \<project-root>  refers to this projects root directory.

```
cd <project-root>
python ./src/run.py -t -m u_net
### Wait for training to finish
python ./src/run.py -p -m u_net
### File csv output will be available under: ./trained_models/u_net/<last-entry>/submission_u_net_<timestamp>.csv
```

In case this does not run, please read further details below.

## Project structure

Following list describes the directories and their use:

| Folder          | Description                                                  |
| --------------- | ------------------------------------------------------------ |
| assets          | This folder contains all the data needed to train and test the model. |
| report          | Contains the code to generate the projects report.           |
| src             | This folder contains the projects source code.               |
| trained_model   | This folder is automatically generated during the training of the model and will contain the projects checkpoints and the final submission.csv |
| \<project-root> | Contains this README file. All code needs to be executed based on this path.<br />Additionally holds the report.pdf and the declaration_of_originality.pdf |

## Prerequisites

In order to run this project following requirements need to be met.

## Environment Setup

This project is based on Python 3.6 and Tensorflow (tested with version 1.7 and 1.8). Therefore the environment needs to be set up with following packages:

- `numpy`
- `matplotlib`
- `keras` (dependent on `tensorflow` and `h5py`)
- `tensorflow` (tested with version 1.8)
- `pandas` 
- Pillow for `PIL`

# Training the model

The score on Kaggle was achieved by training the model until the project automatically finished improving the validation error.

### Usage
```
usage: run.py [-h] [-m {cnn_lr_d,u_net,u_net_dropout}] [-t] [-tr]
              [-d DATA] [-p] [-vis VISUALIZE]

Control program to launch all actions related to this project.

optional arguments:
  -h, --help            show this help message and exit
  -m {cnn_lr_d,u_net,u_net_dropout}, --model {cnn_lr_d,u_net,u_net_dropout}
                        the CNN model to be used, defaults to u_net_dropout
  -t, --train           train the given CNN
  -tr, --train_resume   continue training the given CNN
  -d DATA, --data DATA  path to the data to use (prediction)
  -p, --predict         predict on a test set given the CNN
  -vis VISUALIZE, --visualize VISUALIZE
                        visualize prediction of an image given its id
```

## Executing the training

To start the training execute following command:

``` python <project-root>/src/run.py -t -m u_net  ``` 

**<span style="color:red">Important: The project needs to be executed out of the project-root folder. E.g. the current directory must be the project-root!</span>**

## Executing the prediction

To start the prediction, execute following command:

``` python <project-root>/src/run.py -p -m u_net ```

The output is generated under:

\<project-root>/trained_models/u_net/\<start time training>/submission_u_net_\<timestamp>.csv

## Visualizing the predictions

In order to visualize the predictions, please first make sure to have first generated the submission.csv. Then, just digit:

``` python <project-root>/src/run.py -v <id-number-test-image> -m u_net ```
