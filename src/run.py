#!/usr/bin/env python3 -W ignore::DeprecationWarning

import os, sys
import argparse, glob
import logging

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

import utility
import tests
from models import cnn_lr_d

# Remove tensorflow CPU instruction information.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

###########################################################################################
# USAGE: argument parser
###########################################################################################
parser = argparse.ArgumentParser(description="Control program to launch all actions related to"
                                 " this project.")

parser.add_argument("-m", "--model", action="store",
                    choices=["naive"],
                    default="naive",
                    type=str,
                    help="The CNN model to be used, defaults to naive")
parser.add_argument("-g", "--augment",
                    help="augment training image set",
                    action="store_true")
parser.add_argument("-t", "--train",
                    help="train the given CNN",
                    action="store_true")
parser.add_argument("-r", "--run",
                    help="run a trained version of a given CNN",
                    action="store_true")
parser.add_argument("-c",
                    help="run code tests, can be run only with unittest additional optional"
                    " arguments",
                    action="store_true")


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

args = parser.parse_args()

###########################################################################################
# LOGGER: Setup
###########################################################################################
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
if args.very_verbose:
    console.setLevel(logging.DEBUG)
elif args.verbose:
    console.setLevel(logging.INFO)
elif not args.quiet:
    console.setLevel(logging.WARNING)
else:
    console.setLevel(logging.ERROR)

logger.addHandler(console)
logger.addHandler(logfile)

###########################################################################################
# RUN.PY: action implementation
###########################################################################################
if args.c:
    # Run code tests and exit
    logger.info("Running tests ...")
    console.setLevel(logging.WARNING)
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
    if args.model == "naive":
        model = cnn_lr_d.Model(os.path.join(os.path.dirname(file_path),
                                            os.path.normpath("assets/training/data")))
        model.train(not args.quiet)
        model.save("first_test.h5")

if args.run:
    # Test CNN model
    logger.warning("Requires training")
