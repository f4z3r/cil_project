#!/usr/bin/env python3 -OO

"""Contains all the common data types used in the project."""

import os
import sys

properties = {
    "OUTPUT_DIR": None,
    "SRC_DIR": os.path.dirname(sys.modules['__main__'].__file__),
    "TRAIN_DIR_400": os.path.join(os.path.dirname(sys.modules['__main__'].__file__),
                              os.path.normpath("../assets/400/training")),
    "VAL_DIR_400": os.path.join(os.path.dirname(sys.modules['__main__'].__file__),
                            os.path.normpath("../assets/400/validation")),
    "TRAIN_DIR_608": os.path.join(os.path.dirname(sys.modules['__main__'].__file__),
                              os.path.normpath("../assets/608/training")),
    "TEST_DIR": os.path.join(os.path.dirname(sys.modules['__main__'].__file__),
                             os.path.normpath("../assets/tests")),
    "TESTING_DIR": os.path.join(os.path.dirname(sys.modules['__main__'].__file__),
                                os.path.normpath("../assets/testing")),
    "LOG_DIR": None,

}
