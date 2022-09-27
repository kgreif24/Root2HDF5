""" make_set.py - This program will generate an .h5 file ready for training,
starting from intermediates file. It has been refactored to use the
SetBuilder class.

Author: Kevin Greif
Last updated 9/26/22
python3
"""

from set_builder import SetBuilder

build_dict = {
    'signal': '/DFS-L/DATA/whiteson/kgreif/JetTaggingH5/int_test_esup/',
    'background': '/DFS-L/DATA/whiteson/kgreif/JetTaggingH5/int_test_nominal/',
    'test_name': '/DFS-L/DATA/whiteson/kgreif/JetTaggingH5/syst_sets/test.h5',
    'train_name': '/DFS-L/DATA/whiteson/kgreif/JetTaggingH5/syst_sets/train.h5',
    'test_frac': 0.2,
    'stack': True
}

sb = SetBuilder(build_dict)
sb.run()
