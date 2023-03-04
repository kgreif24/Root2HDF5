""" make_set.py - This program will generate an .h5 file ready for training,
starting from intermediates file. It has been refactored to use the
SetBuilder class.

Author: Kevin Greif
Last updated 9/26/22
python3
"""

from set_builder import SetBuilder
import processing_utils as pu
import preprocessing as pp

build_dict = {
    'signal': '/DFS-L/DATA/whiteson/kgreif/JetTaggingH5/int_zprime_tst_nominal/',
    'background': None,
    'n_files': -1,
    'test_name': '/DFS-L/DATA/whiteson/kgreif/JetTaggingH5/test.h5',
    'train_name': '/DFS-L/DATA/whiteson/kgreif/JetTaggingH5/train.h5',
    'test_frac': 0.1,
    'stack_constits': True,
    'stack_jets': True,
    'jet_fields': ['_pt', '_eta', '_phi', '_m'],
    'jet_keys': ['jet', 'jet_Pileup', 'jet_JES', 'jet_true'],
    'weight_func': None,
    'standards': False
}

sb = SetBuilder(build_dict)
sb.run()
