#!/usr/bin/env python3

import argparse, glob
import logging


import utility


###########################################################################################
# USAGE: argument parser
###########################################################################################
parser = argparse.ArgumentParser(description="Control program to launch all actions related to"
                                 " this project.")
parser.add_argument("-v", "--verbose",
                    help="provide verbose output",
                    action="store_true")
parser.add_argument("model", action="store",
                    choices=["naive"],
                    default="naive",
                    type=str,
                    help="The CNN model to be used")

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("-g", "--augment",
                   help="augment training image set",
                   action="store_true")
group.add_argument("-t", "--train",
                   help="train the given CNN",
                   action="store_true")
group.add_argument("-a", "--all",
                   help="augment, train and run test for the given CNN",
                   action="store_true")
group.add_argument("-u", "--train_run",
                   help="train and run test for the given CNN",
                   action="store_true")
group.add_argument("-r", "--run",
                   help="run tests of a given CNN",
                   action="store_true")

args = parser.parse_args()

###########################################################################################
# LOGGER: Setup
###########################################################################################
logger = logging.getLogger("cil_project")
logger.setLevel(logging.DEBUG)
console = logging.StreamHandler()
logfile = logging.FileHandler("../logs/run.log", 'a')
console_formatter = logging.Formatter("%(message)s")
logfile_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

console.setFormatter(console_formatter)
logfile.setFormatter(logfile_formatter)

logfile.setLevel(logging.WARNING)
if args.verbose:
    console.setLevel(logging.INFO)
else:
    console.setLevel(logging.WARNING)

logger.addHandler(console)
logger.addHandler(logfile)


###########################################################################################
# RUN.PY: action implementation
###########################################################################################
if args.all or args.augment:
    # Augment data set
    if len(glob.glob("../assets/training/data/*.png")) == 100:
        logger.info("Augmenting training data ...")
        utility.augment_img_set("../assets/training/data")
    else:
        logger.warning("Skipped. Please ensure only the 100 original images are contained in the"
                       " `assets/training/data` folder")

    if len(glob.glob("../assets/training/verify/*.png")) == 100:
        logger.info("Augmenting training verification data ...")
        utility.augment_img_set("../assets/training/verify")
    else:
        logger.warning("Skipped. Please ensure only the 100 original images are contained in the"
                       " `assets/training/data` folder")



if args.all or args.train or args.train_run:
    # Train CNN model
    pass

if args.all or args.run or args.train_run:
    # Test CNN model
    pass
