#!/usr/bin/env python3 -W ignore::DeprecationWarning

import os, sys
import argparse, glob
import logging

import warnings
from generators.PatchImageGenerator import PatchImageGenerator

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
    parser.add_argument("-g", "--augment",
                        help="augment training image set",
                        action="store_true")
    parser.add_argument("-d", "--validation_set",
                        help="create validation set from training set",
                        action="store_true")
    parser.add_argument("-t", "--train",
                        help="train the given CNN",
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

    import utility
    import tests
    from models import cnn_lr_d, cnn_model

    if args.command == "check":
        # Run code tests and exit
        logger.info("Running tests ...")
        logger.handlers[0].setLevel(logging.WARNING)
        sys.argv[1:] = sys.argv[2:]
        tests.run()
        sys.exit(0)

    if args.augment:
        # Augment data set
        if len(glob.glob(os.path.join(file_path, "../assets/training/data/*.png"))) <= 100:
            logger.info("Augmenting training data ...")
            utility.augment_img_set(os.path.join(file_path,
                                                 os.path.normpath("../assets/training/data")))
        else:
            logger.warning("Skipped. Please ensure only the 100 original images are contained in the"
                           " `assets/training/data` folder")

        if len(glob.glob(os.path.join(file_path, "../assets/training/verify/*.png"))) <= 100:
            logger.info("Augmenting training verification data ...")
            utility.augment_img_set(os.path.join(file_path,
                                                 os.path.normpath("../assets/training/verify")))
        else:
            logger.warning("Skipped. Please ensure only the 100 original images are contained in the"
                           " `assets/training/data` folder")

    if args.validation_set:
        if len(glob.glob(os.path.join(file_path, "../assets/validation/data/*.png"))) != 0 or \
                len(glob.glob(os.path.join(file_path, "../assets/validation/verify/*.png"))) != 0:
            logging.warning("Skipped validation set generation, make sure no validation set"
                            " is already present.")
        else:
            logger.info("Creating validation data ...")
            utility.get_validation_set(os.path.join(file_path,
                                                    os.path.normpath("../assets/training/data")))

    if args.train:
        if args.model == "cnn_lr_d":
            train_generator = PatchImageGenerator(os.path.normpath("../assets/training/data"), os.path.normpath("../assets/training/verify"))
            validation_generator = PatchImageGenerator(os.path.normpath("../assets/validation/data"), os.path.normpath("../assets/validation/verify"))

            model = cnn_lr_d.CnnLrD(train_generator, validation_generator)
            model.train(not args.quiet)
            model.save("first_test.h5")

        elif args.model == "cnn_model":
            model = cnn_model.CNN_keras(os.path.join(os.path.dirname(file_path), os.path.normpath("assets/training/data")),
                                        os.path.join(os.path.dirname(file_path), os.path.normpath("assets/validation/data/")))
            model.train(not args.quiet)
            model.save("first_test.h5")

    if args.run:
        # Test CNN model
        logger.warning("Requires training")
