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
    'n_files': 15,
    'test_name': '/DFS-L/DATA/whiteson/kgreif/JetTaggingH5/train_s2_ln_small.h5',
    'train_name': '/DFS-L/DATA/whiteson/kgreif/JetTaggingH5/test_s2_ln_small.h5',
    'test_frac': 0.1,
    'stack_constits': True,
    'stack_jets': False,
    'jet_fields': ['_px', '_py', '_pz', '_pt', '_E'],
    'jet_keys': ['jet', 'jet_Pileup', 'jet_JES', 'jet_true'],
    'weight_func': pu.flat_weights,
    'standards': False
}

sb = SetBuilder(build_dict)
sb.run()
