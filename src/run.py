#!/usr/bin/env python3

import os, sys
import argparse, glob
import logging


import utility


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


verbosity_group = parser.add_mutually_exclusive_group()
verbosity_group.add_argument("-v", "--verbose",
                    help="provide verbose output",
                    action="store_true")
verbosity_group.add_argument("-vv", "--very_verbose",
                    help="provide even more verbose output",
                    action="store_true")

args = parser.parse_args()

###########################################################################################
# LOGGER: Setup
###########################################################################################
file_path = os.path.dirname(os.path.abspath(__file__))

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
else:
    console.setLevel(logging.WARNING)

logger.addHandler(console)
logger.addHandler(logfile)

###########################################################################################
# RUN.PY: action implementation
###########################################################################################
if args.augment:
    # Augment data set
    if len(glob.glob("../assets/training/data/*.png")) == 100:
        logger.info("Augmenting training data ...")
        utility.augment_img_set(os.path.join(file_path, "../assets/training/data"))
    else:
        logger.warning("Skipped. Please ensure only the 100 original images are contained in the"
                       " `assets/training/data` folder")

    if len(glob.glob("../assets/training/verify/*.png")) == 100:
        logger.info("Augmenting training verification data ...")
        utility.augment_img_set(os.path.join(file_path, "../assets/training/verify"))
    else:
        logger.warning("Skipped. Please ensure only the 100 original images are contained in the"
                       " `assets/training/data` folder")



if args.train:
    # Train CNN model
    logger.warning("Training is not implemented yet")

if args.run:
    # Test CNN model
    logger.warning("Requires training")
