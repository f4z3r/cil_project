#!/usr/bin/env python3

import argparse

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

if args.all or args.augment:
    # Augment data set
    pass

if args.all or args.train or args.train_run:
    # Train CNN model
    pass

if args.all or args.run or args.train_run:
    # Test CNN model
    pass
