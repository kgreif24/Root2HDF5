""" r2h_raw.py - This script runs the r2h step of data processing, applying
minimal preprocessing. For generating .h5 for jet calibrations.

Author: Kevin Greif
python3
Last updated 9/13/22
"""

from root_converter import RootConverter
import processing_utils as pu
import preprocessing as pp

# Define convert_dict which is passed to RootConverter class
convert_dict = {
    'tree_name': ':IsolatedJet_tree',
    'trim': True,
    'source_list': './dat/calib_test.list',
    'flatten': True,
    'max_constits': 200,
    'target_dir': './int_calib/',
    'name_stem': 'calib_data',
    'n_targets': 1,
    'total': 4000000,
    'constit_func': pp.raw_preprocess,
    'cut_func': None,
    's_constit_branches': [
        'jet_constit_pt', 'jet_constit_eta',
        'jet_constit_phi'
    ],
    'test_name': 'jet_pt',
    'pt_name': 'jet_constit_pt',
    't_constit_branches': [
        'jet_constit_pt', 'jet_constit_eta', 'jet_constit_phi'
    ],
    'jet_branches': [
        'jet_pt', 'jet_eta', 'jet_phi', 'jet_E',
        'jet_true_pt', 'jet_true_eta', 'jet_true_phi', 'jet_true_e',
    ],
    'event_branches': [
        'eventNumber', 'rho', 'NPV', 'actualInteractionsPerCrossing'
    ],
    'cut_branches': []
}

# Build the class
rc = RootConverter(convert_dict)

# Run main program
rc.run()
