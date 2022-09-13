""" make_set.py - This program will generate an .h5 file ready for training,
starting from intermediates file. It has been refactored to use the
SetBuilder class.

Author: Kevin Greif
Last updated 8/22/22
python3
"""

from set_builder import SetBuilder

build_dict = {
    'signal': './dataloc/transferLearning/int_zprime/',
    'background': './dataloc/transferLearning/int_dijet/',
    'test_name': None,
    'train_name': './dataloc/transferLearning/delphes_zprime_dijet.h5',
    'test_frac': 0
}

sb = SetBuilder(build_dict)
sb.run()
