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
    'signal': '/DFS-L/DATA/whiteson/kgreif/JetTaggingH5/int_zprime_raw_nominal/',
    'background': '/DFS-L/DATA/whiteson/kgreif/JetTaggingH5/int_dijet_raw_nominal/',
    'n_files': -1,
    'test_name': '/DFS-L/DATA/whiteson/kgreif/JetTaggingH5/test_raw_nominal.h5',
    'train_name': '/DFS-L/DATA/whiteson/kgreif/JetTaggingH5/train_raw_nominal.h5',
    'test_frac': 0.1,
    'stack_constits': False,
    'stack_jets': False,
    'jet_fields': ['_pt', '_eta', '_phi', '_m'],
    'jet_keys': ['fjet'],
    'weight_func': pu.match_weights,
    'standards': False
}

sb = SetBuilder(build_dict)
sb.run()
