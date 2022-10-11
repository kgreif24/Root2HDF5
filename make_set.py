""" make_set.py - This program will generate an .h5 file ready for training,
starting from intermediates file. It has been refactored to use the
SetBuilder class.

Author: Kevin Greif
Last updated 9/26/22
python3
"""

from set_builder import SetBuilder
import processing_utils as pu

build_dict = {
    'signal': '/DFS-L/DATA/whiteson/kgreif/JetTaggingH5/int_zprime_s2_ln/',
    'background': '/DFS-L/DATA/whiteson/kgreif/JetTaggingH5/int_dijet_s2_ln/',
    'test_name': '/DFS-L/DATA/whiteson/kgreif/JetTaggingH5/test_s2_ln.h5',
    'train_name': '/DFS-L/DATA/whiteson/kgreif/JetTaggingH5/train_s2_ln.h5',
    'test_frac': 0.1,
    'stack': True,
    'weight_func': pu.match_weights
}

sb = SetBuilder(build_dict)
sb.run()
