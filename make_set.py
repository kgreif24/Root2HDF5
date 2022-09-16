""" make_set.py - This program will generate an .h5 file ready for training,
starting from intermediates file. It has been refactored to use the
SetBuilder class.

Author: Kevin Greif
Last updated 8/22/22
python3
"""

from set_builder import SetBuilder

build_dict = {
    'signal': '/DFS-L/DATA/whiteson/kgreif/JetCalib/h5data/intermediates/',
    'background': None,
    'test_name': '/DFS-L/DATA/whiteson/kgreif/JetCalib/h5data/test.h5',
    'train_name': '/DFS-L/DATA/whiteson/kgreif/JetCalib/h5data/train.h5',
    'test_frac': 0.2
}

sb = SetBuilder(build_dict)
sb.run()
