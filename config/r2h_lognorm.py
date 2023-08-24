""" r2h_lognorm.py - This script runs the r2h step of data processing, applying
the lognorm preprocessing. This is preprocessing used for generating training
data sets.

Author: Kevin Greif
python3
Last updated 10/1/22
"""

from root_converter import RootConverter
import processing_utils as pu
import preprocessing as pp
import syst_variations as syst

# Define convert_dict which is passed to RootConverter class
convert_dict = {
    'tree_name': ':FlatSubstructureJetTree',
    'trim': True,
    'source_list': './dat/zprime_data.list',
    'max_constits': 200,
    'mask_lim': 100,
    'unit_multiplier': 1.0,
    'target_dir': './dataloc/int_zprime_tst_nominal/',
    'name_stem': 'zprime_tst_nominal_',
    'n_targets': 100,
    'total': 5000000,
    # 'total': 20000000,
    'flatten': False,
    'constit_func': pp.train_preprocess,
    'syst_func': None,
    'cut_func': pu.signal_cuts,
    'jet_func': None,
    's_constit_branches': [
        'fjet_clus_pt', 'fjet_clus_eta',
        'fjet_clus_phi', 'fjet_clus_E',
        'fjet_clus_taste'
    ],
    'pt_name': 'fjet_clus_pt',
    'event_branches': [],
    't_constit_branches': [
        'fjet_clus_eta', 'fjet_clus_phi', 'fjet_clus_log_pt', 'fjet_clus_log_E',
        'fjet_clus_lognorm_pt', 'fjet_clus_lognorm_E', 'fjet_clus_dR',
        'fjet_clus_taste'
    ],
    's_jet_branches': ['fjet_pt', 'fjet_eta', 'fjet_phi', 'fjet_m'],
    't_jet_branches': ['fjet_pt', 'fjet_eta', 'fjet_phi', 'fjet_m'],
    'images_branch': [],
    'cut_branches': [
        'fjet_truthJet_eta', 'fjet_truthJet_pt', 'fjet_numConstituents', 'fjet_m',
        'fjet_truth_dRmatched_particle_flavor', 'fjet_truth_dRmatched_particle_dR',
        'fjet_truthJet_dRmatched_particle_dR_top_W_matched', 'fjet_ungroomed_truthJet_m',
        'fjet_truthJet_ungroomedParent_GhostBHadronsFinalCount', 'fjet_ungroomed_truthJet_Split23',
        'fjet_ungroomed_truthJet_pt'
    ],
    'weight_branches': []
}

# Build the class
rc = RootConverter(convert_dict)

# Run main programs
rc.run()
