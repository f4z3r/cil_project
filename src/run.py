#!/usr/bin/env python3 -W ignore::DeprecationWarning

import os, sys
import argparse, glob
import logging

import warnings
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
    parser_c = subparsers.add_parser("check", help="Test code")

    parser.add_argument("-m", "--model", action="store",
                        choices=["cnn_lr_d", "dnn_class"],
                        default="cnn_lr_d",
                        type=str,
                        help="the CNN model to be used, defaults to cnn_lr_d")
    parser.add_argument("-g", "--augment",
                        help="augment training image set",
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
# Functions for jupyter notebook


# This is not required

# Simply write `%run run.py -t -v` to train network in the jupyter notebook.
###########################################################################################

def train(model, verbosity=1):
    """Train model `model`.

    Args:
        model (str): the model to train. Use `run.py -h` to see available models.
        verbosity (int): default=1 - verbosity of output
    """
    import utility
    import tests
    from models import cnn_lr_d

    # Initialise logger
    logger = _setup_logger()

    verbose = False
    if verbosity > 0:
        verbose = True
        logger.handlers[0].setLevel(logging.INFO)
    elif verbosity > 1:
        logger.handlers[0].setLevel(logging.DEBUG)
    else:
        logger.handlers[0].setLevel(logging.WARNING)

    if model == "naive":
        model = cnn_lr_d.Model(os.path.join(os.path.dirname(file_path),
                                            os.path.normpath("assets/training/data")))
        model.train(verbose)
        model.save("google_colab_test.h5")


###########################################################################################
# RUN.PY actions.
###########################################################################################
if __name__ == "__main__":
    file_path = os.path.dirname(os.path.abspath(__file__))

    args = _setup_argparser()
    logger = _setup_logger(args)

    import utility
    import tests
    from models import cnn_lr_d

    if args.command == "check":
        # Run code tests and exit
        logger.info("Running tests ...")
        logger.handlers[0].setLevel(logging.WARNING)
        sys.argv[1:] = sys.argv[2:]
        tests.run()
        sys.exit(0)


    if args.augment:
        # Augment data set
        if len(glob.glob(os.path.join(file_path, "../assets/training/data/*.png"))) == 100:
            logger.info("Augmenting training data ...")
            utility.augment_img_set(os.path.join(file_path, "../assets/training/data"))
        else:
            logger.warning("Skipped. Please ensure only the 100 original images are contained in the"
                           " `assets/training/data` folder")

        if len(glob.glob(os.path.join(file_path, "../assets/training/verify/*.png"))) == 100:
            logger.info("Augmenting training verification data ...")
            utility.augment_img_set(os.path.join(file_path, "../assets/training/verify"))
        else:
            logger.warning("Skipped. Please ensure only the 100 original images are contained in the"
                           " `assets/training/data` folder")

    if args.train:
        if args.model == "cnn_lr_d":
            model = cnn_lr_d.Model(os.path.join(os.path.dirname(file_path),
                                                os.path.normpath("assets/training/data")))
            model.train(not args.quiet)
            model.save("first_test.h5")
        elif args.model == "dnn_class":
            logger.warning("Not fully implemented")

    if args.run:
        # Test CNN model
        logger.warning("Requires training")
