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
    'source_list': './dat/calib_data.list',
    'flatten': True,
    'max_constits': 80,
    'mask_lim': 0,
    'unit_multiplier': 1.0,
    'target_dir': './dataloc/h5data/intermediates/',
    'name_stem': 'calib_data',
    'n_targets': 1,
    'total': 10000,
    'constit_func': pp.cartesian_pt_preprocess,
    'jet_func': pp.jet_preprocess,
    'cut_func': None,
    'syst_func': None,
    's_constit_branches': [
        'jet_constit_pt', 'jet_constit_eta',
        'jet_constit_phi'
    ],
    'test_name': 'jet_pt',
    'pt_name': 'jet_constit_pt',
    't_constit_branches': [
        'jet_constit_px', 'jet_constit_py', 'jet_constit_pz',
        'jet_constit_pt'
    ],
    'images_branch': [],
    's_jet_branches': [
        'jet_pt', 'jet_eta', 'jet_phi', 'jet_E',
        'jet_PileupPt', 'jet_PileupEta', 'jet_PileupPhi', 'jet_PileupE',
        'jet_JESPt', 'jet_JESEta', 'jet_JESPhi', 'jet_JESE',
        'jet_true_pt', 'jet_true_eta', 'jet_true_phi', 'jet_true_e',
    ],
    't_jet_branches': [
        'jet_px', 'jet_py', 'jet_pz', 'jet_pt', 'jet_E',
        'jet_Pileup_px', 'jet_Pileup_py', 'jet_Pileup_pz', 'jet_Pileup_pt', 'jet_Pileup_E',
        'jet_JES_px', 'jet_JES_py', 'jet_JES_pz', 'jet_JES_pt', 'jet_JES_E',
        'jet_true_px', 'jet_true_py', 'jet_true_pz', 'jet_true_pt', 'jet_true_E'
    ],
    'event_branches': [
        'eventNumber', 'rho', 'NPV', 'actualInteractionsPerCrossing'
    ],
    'cut_branches': [],
    'weight_branches': [],
    'nn_weights': None
}

# Build the class
rc = RootConverter(convert_dict)

# Run main program
rc.run()
