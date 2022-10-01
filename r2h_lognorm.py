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
    'source_list': './dat/Zprime_taste.list',
    'rw_type': 'w',
    'max_constits': 200,
    'target_dir': './dataloc/int_nominal/',
    'n_targets': 8,
    # 'total': 22375114,
    'total': 4000000,
    'constit_func': pp.raw_preprocess,
    'syst_func': syst.energy_scale,
    'cut_func': pu.signal_cuts,
    's_constit_branches': [
        'fjet_clus_pt', 'fjet_clus_eta',
        'fjet_clus_phi', 'fjet_clus_E',
        'fjet_clus_taste'
    ],
    'pt_name': 'fjet_clus_pt',
    'hl_branches': [
        'fjet_Tau1_wta', 'fjet_Tau2_wta', 'fjet_Tau3_wta',
        'fjet_Tau4_wta', 'fjet_Split12', 'fjet_Split23',
        'fjet_ECF1', 'fjet_ECF2', 'fjet_ECF3', 'fjet_C2',
        'fjet_D2', 'fjet_Qw', 'fjet_L2', 'fjet_L3',
        'fjet_ThrustMaj'
    ],
    't_constit_branches': [
        'fjet_clus_eta', 'fjet_clus_phi', 'fjet_clus_pt', 'fjet_clus_E',
        'fjet_clus_taste'
    ],
    'jet_branches': ['fjet_pt', 'fjet_eta', 'fjet_phi', 'fjet_m'],
    'cut_branches': [
        'fjet_truthJet_eta', 'fjet_truthJet_pt', 'fjet_numConstituents', 'fjet_m',
        'fjet_truth_dRmatched_particle_flavor', 'fjet_truth_dRmatched_particle_dR',
        'fjet_truthJet_dRmatched_particle_dR_top_W_matched', 'fjet_ungroomed_truthJet_m',
        'fjet_truthJet_ungroomedParent_GhostBHadronsFinalCount', 'fjet_ungroomed_truthJet_Split23',
        'fjet_ungroomed_truthJet_pt'
    ]
}

# Build the class
rc = RootConverter(convert_dict)

# Run main programs
rc.run(direction='up')

