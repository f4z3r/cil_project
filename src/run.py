#!/usr/bin/env python3 -W ignore::DeprecationWarning

import argparse
import logging
import os
import sys
import warnings

import tests
from keras.models import load_model
from generators.PatchTrainImageGenerator import PatchTrainImageGenerator
from generators.PatchTestImageGenerator import PatchTestImageGenerator
from models import cnn_lr_d, cnn_model
from models import predictions

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

# Remove tensorflow CPU instruction information.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def _setup_argparser():
    """Sets up the argument parser and returns the arguments.

    Returns:
        argparse.Namespace: The command line arguments.
    """
    parser = argparse.ArgumentParser(description="Control program to launch all actions related to"
                                                 " this project.")

    verbosity_group = parser.add_mutually_exclusive_group()
    verbosity_group.add_argument("-v", "--verbose",
                                 help="provide verbose output",
                                 action="store_true")
    verbosity_group.add_argument("-vv", "--very_verbose",
                                 help="provide even more verbose output",
                                 action="store_true")
    verbosity_group.add_argument("-q", "--quiet",
                                 help="provide next to no output to console",
                                 action="store_true")

    subparsers = parser.add_subparsers(dest="command", help="Test utilities")
    parser_c = subparsers.add_parser("check",
                                     help="Run unittest.main, accepts unittest options.")
    parser_c.add_argument("tests",
                          help="a list of any number of test modules, classes and test methods.",
                          nargs="*")
    parser_c.add_argument("-v", "--verbose",
                          help="Verbose Output",
                          action="store_true")
    parser_c.add_argument("-q", "--quiet",
                          help="Quiet Output",
                          action="store_true")
    parser_c.add_argument("--locals",
                          help="Show local variables in tracebacks",
                          action="store_true")
    parser_c.add_argument("-f", "--failfast",
                          help="Stop on first fail or error",
                          action="store_true")
    parser_c.add_argument("-c", "--catch",
                          help="Catch Ctrl-C and display results so far",
                          action="store_true")
    parser_c.add_argument("-b", "--buffer",
                          help="Buffer stdout and stderr during tests",
                          action="store_true")
    parser_c.add_argument("-p", "--pattern",
                          help="Pattern to match tests ('test*.py' default)")

    parser.add_argument("-m", "--model", action="store",
                        choices=["cnn_lr_d", "cnn_model"],
                        default="cnn_lr_d",
                        type=str,
                        help="the CNN model to be used, defaults to cnn_lr_d")
    parser.add_argument("-t", "--train",
                        help="train the given CNN",
                        action="store_true")
    parser.add_argument("-pr", "--predict",
                        help="predict on a test set given the CNN",
                        action="store_true")
    parser.add_argument("-ptm", "--path_to_trained_model",
                        help="path to load a specific trained model",
                        action="store_true")
    parser.add_argument("-r", "--run",
                        help="run a trained version of a given CNN",
                        action="store_true")

    args, unknown = parser.parse_known_args()

    return args


def _setup_logger(args=None):
    """Set up the logger.

    Args:
        args (argparse.Namespace): the command line arguments from runnning the file.

    Returns:
        logging.Logger: A logger.
    """
    file_path = os.path.dirname(os.path.abspath(__file__))

    try:
        os.mkdir(os.path.join(file_path, "../logs"))
    except OSError:
        pass

    logger = logging.getLogger("cil_project")
    logger.setLevel(logging.DEBUG)
    console = logging.StreamHandler()
    logfile = logging.FileHandler(os.path.join(file_path, "../logs/run.log"), 'a')
    console_formatter = logging.Formatter("%(message)s")
    logfile_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    console.setFormatter(console_formatter)
    logfile.setFormatter(logfile_formatter)

    logfile.setLevel(logging.WARNING)
    if args is None:
        console.setLevel(logging.INFO)
    elif args.very_verbose:
        console.setLevel(logging.DEBUG)
    elif args.verbose:
        console.setLevel(logging.INFO)
    elif not args.quiet:
        console.setLevel(logging.WARNING)
    else:
        console.setLevel(logging.ERROR)

    logger.addHandler(console)
    logger.addHandler(logfile)

    return logger


###########################################################################################
# RUN.PY actions.
###########################################################################################
if __name__ == "__main__":
    file_path = os.path.dirname(os.path.abspath(__file__))

    args = _setup_argparser()
    logger = _setup_logger(args)

    if args.command == "check":
        # Run code tests and exit
        logger.info("Running tests ...")
        logger.handlers[0].setLevel(logging.WARNING)
        sys.argv[1:] = sys.argv[2:]
        tests.run()
        sys.exit(0)

    if args.train:
        if args.model == "cnn_lr_d":
            train_generator = PatchTrainImageGenerator(os.path.normpath("../assets/training/data"),
                                                  os.path.normpath("../assets/training/verify"))
            validation_generator = PatchTrainImageGenerator(os.path.normpath("../assets/validation/data"),
                                                       os.path.normpath("../assets/validation/verify"))

            model = cnn_lr_d.CnnLrD(train_generator, validation_generator)
            model.train(not args.quiet)
            model.save("first_test.h5")

        elif args.model == "cnn_model":
            train_generator = PatchTrainImageGenerator(os.path.normpath("../assets/training/data"),
                                                  os.path.normpath("../assets/training/verify"))
            validation_generator = PatchTrainImageGenerator(os.path.normpath("../assets/validation/data"),
                                                       os.path.normpath("../assets/validation/verify"))
            model = cnn_model.CNN_keras(train_generator, validation_generator)
            model.train(not args.quiet)
            model.save("first_test.h5")

    if args.predict:
        #TODO complete this part -> Wait Jakob to create directories with saved files
        
        """trained_models_dir = os.path.normpath("../trained_models/args.model/")
        if not os.path.exists(trained_models_dir):
            os.makedirs(trained_models_dir)"""

        if args.path_to_trained_model:
            path_to_trained_model = args.path_to_trained_model
        else:
            path_to_trained_model = os.path.normpath("../trained_models/"+args.model+"/")
            all_runs = [os.path.join(path_to_trained_model, o) for o in os.listdir(path_to_trained_model)
                        if os.path.isdir(os.path.join(path_to_trained_model, o))]
            latest_run = max(all_runs, key=os.path.getmtime)  # get the latest run
            checkpoint_path = path_to_trained_model+latest_run+"/"
            path_model_to_restore = checkpoint_path +"*.h5" #TODO + name of the saved model to fetch
            print("Loading the last checkpoint of the model ",args.model," from: ",checkpoint_path)


        if args.model == "cnn_lr_d":
            
            test_generator = PatchTestImageGenerator(os.path.normpath("../assets/testing/data"),
                                                  os.path.normpath("../assets/testing/predictions"))
            
            restored_model = load_model(path_model_to_restore)
            model = predictions.Prediction_model(test_generator = test_generator, restored_model = restored_model)
            model.prediction_given_model()
        
        elif args.model == "cnn_model":
            test_generator = PatchTestImageGenerator(os.path.normpath("../assets/testing/data"),
                                                  os.path.normpath("../assets/testing/predictions"))
            restored_model = load_model(path_model_to_restore)
            model = predictions.Prediction_model(test_generator = test_generator, restored_model = restored_model)
            model.prediction_given_model()
            
    if args.run:
        # Test CNN model
        logger.warning("Requires training or predicting")
