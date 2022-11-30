""" preprocessing.py - This script will define functions that can be used
to pre-preprocess the constituent level information coming from the raw
nTuples. Different pre-processing schemes can be applied easily by
calling different functions from this module.

Author: Kevin Greif
Last updated 6/29/22
python3
"""

import numpy as np
import awkward as ak


def raw_preprocess(jets, sort_indeces, zero_indeces, params):
    """ raw_preprocess - This preprocessing function applies to minimal
    preprocessing used in the public facing data sets.

    args and returns are standard. Set params['logs'] to true to take logarithm
    of dimensionful inputs (pT or energy)
    """

    # Initialize preprocess dict
    preprocess = {}

    # Loop through target constituents and onehots
    loop_branches = params['t_constit_branches'] + params['t_onehot_branches']
    for name in loop_branches:

        # Get branch
        branch = jets[name]

        # If this is energy or pt branch, take logarithm
        if params['logs'] and (('pt' in name) or ('E' in name)):
            branch = np.log(branch)

        # Zero pad
        temp = ak.pad_none(branch, params['max_constits'], axis=1, clip=True)
        temp = ak.to_numpy(ak.fill_none(temp, 0, axis=1))

        # Set small pT constituents to zero
        temp[zero_indeces] = 0

        # We sort by DECREASING pT, which necessitates complicated indexing
        temp = np.take_along_axis(temp, sort_indeces, axis=1)[:,::-1]

        # Write to batch dict
        preprocess[name] = temp

    # Finally we return dict
    return preprocess


def cartesian_pt_preprocess(jets, sort_indeces, zero_indeces, params):
    """ cartesian_pt_preprocess - This preprocessing function converts the
    pt, eta, phi coordinates for jet constituents to px, py, pz. It also places
    the zero padded and sorted pT information in the return dictionary.

    args and returns are standard. Setting zero_indeces to an empty list will
    result in no masking being applied to the constituents, so long as the
    dtype is set to integer 
    """

    # Pull pt, eta, phi of the constituents
    # Naming conventions are hardcoded for jet calib project. Could refactor
    # if needed one day.
    pts = jets['jet_constit_pt']
    eta = jets['jet_constit_eta']
    phi = jets['jet_constit_phi']

    # Calculate cartesian coordinates
    pxs = pts * np.cos(phi)
    pys = pts * np.sin(phi)
    pzs = pts * np.sinh(eta)

    # Send cartesian coordinates to zero padded numpy
    pxs_zero = zero_pad(pxs, params['max_constits'])
    pys_zero = zero_pad(pys, params['max_constits'])
    pzs_zero = zero_pad(pzs, params['max_constits'])
    pts_zero = zero_pad(pts, params['max_constits'])

    # Mask constituents
    pxs_zero[zero_indeces] = 0
    pys_zero[zero_indeces] = 0
    pzs_zero[zero_indeces] = 0
    pts_zero[zero_indeces] = 0

    # Sort constituents
    pxs_sort = np.take_along_axis(pxs_zero, sort_indeces, axis=1)[:,::-1]
    pys_sort = np.take_along_axis(pys_zero, sort_indeces, axis=1)[:,::-1]
    pzs_sort = np.take_along_axis(pzs_zero, sort_indeces, axis=1)[:,::-1]
    pts_sort = np.take_along_axis(pts_zero, sort_indeces, axis=1)[:,::-1]

    # Return dictionary
    return {'jet_constit_px': pxs_sort, 
            'jet_constit_py': pys_sort,
            'jet_constit_pz': pzs_sort,
            'jet_constit_pt': pts_sort}
    

def energy_norm(jets, indeces, max_constits=200, **kwargs):
    """ energy_norm - Defines the standard energy constituent preprocessing,
    where the transverse momentum and energy are sorted by decreasing pt
    and normalized by the sum of the pt of all jet constituents (jet pt). Function
    returns preprocessed arrays for pt and energy.

    Arguments:
    jets (dict): Dictionary whose elements are awkard arrays giving the constituent
    pt and energy. Usually a batch of a loop over a .root file using uproot.
    indeces (array): The indeces which will sort the constituents by INCREASING pt. Sort
    will be reflected to sort by decreasing pt.
    max_constits (int): The number of constituents to keep in our jets. Jets shorter
    than this will be zero padded, jets longer than this will be truncated.

    Returns:
    (array) A zero padded array of the constituent pt
    (array) A zero padded array of the constituent eta
    """

    # Pt (order by pt and normalize)
    pt = jets['fjet_clus_pt']

    # First normalize while still in ak array format
    pt_sum = np.sum(pt, axis=1)
    pt_norm = pt / pt_sum[:,np.newaxis]

    # Zero pad and send to numpy
    pt_zero = ak.pad_none(pt_norm, max_constits, axis=1, clip=True)
    pt_zero = ak.to_numpy(ak.fill_none(pt_zero, 0, axis=None))

    # Sort jets by decreasing pt
    # Odd slicing is needed to invert pt ordering (decreasing vs increasing)
    pt_sort = np.take_along_axis(pt_zero, indeces, axis=1)[:,::-1]


    # E (order by Pt and normalize by Pt)
    en = jets['fjet_clus_E']

    # Normalize by pt (this is not a bug!!)
    en_norm = en / pt_sum[:,np.newaxis]

    # Zero pad and send to numpy
    en_zero = ak.pad_none(en_norm, max_constits, axis=1, clip=True)
    en_zero = ak.to_numpy(ak.fill_none(en_zero, 0, axis=None))

    # Sort by decreasing pt
    en_sort = np.take_along_axis(en_zero, indeces, axis=1)[:,::-1]


    # Now simply return preprocessed pt/energy
    return pt_sort, en_sort


def log_norm(jets, name, sort_indeces, zero_indeces, max_constits=200, **kwargs):
    """ log_norm - As opposed to energy norm, this preprocessing function will
    return the logs of the normalized, and un-normalized pT or E. It is meant
    to recreate the preprocessing used in the Particle Net paper.

    Arguments:
    jets (dict): Dictionary whose elements are awkard arrays giving the constituent
    pt and energy. Usually a batch of a loop over a .root file using uproot.
    name (str): Either 'fjet_clus_pt', or 'fjet_clus_E'. Function will apply preprocessing
    to this element of the jets dict.
    sort_indeces (array): The indeces which will sort the constituents by INCREASING pt. Sort
    will be reflected to sort by decreasing pt.
    zero_indeces (array): The indeces of constituents which do not pass
    minimum pT cut. These are masked to zero.
    max_constits (int): The number of constituents to keep in our jets. Jets shorter
    than this will be zero padded, jets longer than this will be truncated.

    Returns:
    (array) - log(constits) values, in the shape (num_jets, max_constits)
    (array) - log(constits / sum(constits)), in the same shape
    """

    # We must calculate log, lognorm, and order by decreasing
    cons = jets[name]

    # First take log and lognorm while still in ak array format
    log_cons = np.log(cons)
    sum_cons = np.sum(cons, axis=1)
    norm_cons = cons / sum_cons[:,np.newaxis]
    lognorm_cons = np.log(norm_cons)

    # Zero pad and send to numpy
    log_cons_zero = ak.pad_none(log_cons, max_constits, axis=1, clip=True)
    log_cons_zero = ak.to_numpy(ak.fill_none(log_cons_zero, 0, axis=None))
    lognorm_cons_zero = ak.pad_none(lognorm_cons, max_constits, axis=1, clip=True)
    lognorm_cons_zero = ak.to_numpy(ak.fill_none(lognorm_cons_zero, 0, axis=None))

    # Mask small pT constituents
    log_cons_zero[zero_indeces] = 0
    lognorm_cons_zero[zero_indeces] = 0

    # Sort jets by decreasing pt
    # Odd slicing is needed to invert pt ordering (decreasing vs increasing)
    log_cons_sort = np.take_along_axis(log_cons_zero, sort_indeces, axis=1)[:,::-1]
    lognorm_cons_sort = np.take_along_axis(lognorm_cons_zero, sort_indeces, axis=1)[:,::-1]

    # Return results
    return log_cons_sort, lognorm_cons_sort
 

def zero_mass(jets):
    """ zero_mass - A function for setting the constituent energy values to pT * cosh(eta)
    This effectively zeros the mass of all constituents.

    Arguments:
    jets (dict) - Dictionary of awkward arrays containing the constituent level information
    for pT and eta

    Returns:
    (dict) - Derived constituent energy information, packaged in a python dict with the 
    key 'fjet_clus_E'
    """

    # Pull pT and eta
    cons_pt = jets['fjet_clus_pt']
    cons_eta = jets['fjet_clus_eta']

    # Calculate energy
    cons_en = cons_pt * np.cosh(cons_eta)

    return {'fjet_clus_E': cons_en}


def simple_angular(jets, sort_indeces, zero_indeces, max_constits=200, **kwargs):
    """ simple_angular - This function will perform a simple preprocessing
    on the angular constituent information. It is the same preprocessing used
    in 1902.09914.

    Arguments:
    jets (dict): Dictionary whose elements are awkard arrays giving the constituent
    eta and phi. Usually a batch of a loop over a .root file using uproot.
    sort_indeces (array): The indeces which will sort the constituents by INCREASING pt. Sort
    will be reflected to sort by decreasing pt.
    zero_indeces (array): The indeces of constituents which do not pass
    minimum pT cut. These are masked to zero.
    max_constits (int): The number of constituents to keep in our jets. Jets shorter
    than this will be zero padded, jets longer than this will be truncated.

    Returns:
    (array) - A zero padded array giving the preprocessed eta values. It's shape will
    be (num_jets, max_constits)
    (array) - A zero padded array giving the preprocessed phi values.
    """

    # Need to center/rotate/flip constituents BEFORE zero padding.
    eta = jets['fjet_clus_eta']
    phi = jets['fjet_clus_phi']

    # 1. Center hardest constituent in eta/phi plane
    # Find the eta/phi coordinates of hardest constituent in each jet, going to
    # need some fancy indexing
    ax_index = np.arange(0, len(eta), 1)
    first_eta = eta[ax_index, sort_indeces[:,-2]]
    first_phi = phi[ax_index, sort_indeces[:,-1]]

    # Now center
    eta_center = eta - first_eta[:,np.newaxis]
    phi_center = phi - first_phi[:,np.newaxis]

    # Fix discontinuity in phi at +-pi
    phi_center = np.where(phi_center > np.pi, phi_center - 2*np.pi, phi_center)
    phi_center = np.where(phi_center < -np.pi, phi_center + 2*np.pi, phi_center)

    # 2. Rotate such that 2nd hardest constituent sits on negative phi axis
    # Screen indeces for any jets with 1 or 2 constituents (ask about these)
    second_eta = eta_center[ax_index, sort_indeces[:,-2]]
    second_phi = phi_center[ax_index, sort_indeces[:,-2]]
    angle = np.arctan2(second_phi, second_eta) + np.pi/2
    eta_rot = eta_center * np.cos(angle[:,np.newaxis]) + phi_center * np.sin(angle[:,np.newaxis])
    phi_rot = -eta_center * np.sin(angle[:,np.newaxis]) + phi_center * np.cos(angle[:,np.newaxis])

    # 3. If needed, reflect 3rd hardest constituent into positive eta half-plane
    third_eta = eta_rot[ax_index, sort_indeces[:,-3]]
    parity = np.where(third_eta < 0, -1, 1)
    eta_flip = eta_rot * parity[:,np.newaxis]

    # Ready to zero pad eta/phi and send arrays to numpy
    eta_zero = ak.pad_none(eta_flip, max_constits, axis=1, clip=True)
    eta_zero = ak.to_numpy(ak.fill_none(eta_zero, 0, axis=None))
    phi_zero = ak.pad_none(phi_rot, max_constits, axis=1, clip=True)
    phi_zero = ak.to_numpy(ak.fill_none(phi_zero, 0, axis=None))

    # Mask small pT indeces
    eta_zero[zero_indeces] = 0
    phi_zero[zero_indeces] = 0

    # Sort constituents using indeces passed into function
    eta_sort = np.take_along_axis(eta_zero, sort_indeces, axis=1)[:,::-1]
    phi_sort = np.take_along_axis(phi_zero, sort_indeces, axis=1)[:,::-1]

    # Finished preprocessing. Return results
    return eta_sort, phi_sort


def train_preprocess(jets, sort_indeces, zero_indeces, params):
    """ train_preprocess - This function applies the standard preprocessing
    used for training networks. It applies the "lognorm" preprocessing for
    dimensionful (energ and pT) inputs and the "simple_angular" preprocessing
    for coordinate (eta and phi) inputs. It also calculates \DeltaR from
    the preprocessed angular inputs.

    Arguments and returns are standard
    """

    # Apply lognorm preprocessing for pT and energy branches
    log_pt, lognorm_pt = log_norm(
        jets,
        'fjet_clus_pt',
        sort_indeces,
        zero_indeces,
        max_constits=params['max_constits']
    )
    log_en, lognorm_en = log_norm(
        jets,
        'fjet_clus_E',
        sort_indeces,
        zero_indeces,
        max_constits=params['max_constits']
    )

    # Apply simple angular processing for eta and phi branches
    eta, phi = simple_angular(
        jets,
        sort_indeces,
        zero_indeces,
        max_constits=params['max_constits']
    )

    # Calculate \DeltaR with the preprocessed constituents
    dR = np.sqrt(eta**2 + phi**2)

    # Return preprocessed branches
    pp_dict = {
        'fjet_clus_eta': eta, 'fjet_clus_phi': phi, 'fjet_clus_log_pt': log_pt,
        'fjet_clus_log_E': log_en, 'fjet_clus_lognorm_pt': lognorm_pt,
        'fjet_clus_lognorm_E': lognorm_en, 'fjet_clus_dR': dR
    }
    return pp_dict


def zero_pad(ak_data, max_constits, fill=0):
    """ zero_pad - This function converts awkward arrays containing constituent
    data to padded, rectangular numpy arrays.

    Arguments:
    ak_data (ak array) - Awkward array of the constituent data
    max_constits (int) - The number of constituents to include
    fill (float or int) - The value to fill with, usually 0

    Returns:
    (array) - zero padded numpy array
    """

    ak_zero = ak.pad_none(ak_data, max_constits, axis=1, clip=True)
    ak_zero = ak.to_numpy(ak.fill_none(ak_zero, fill, axis=None))

    return ak_zero


def jet_preprocess(jets):
    """ jet_preprocess - This function converts the jet pt, eta,
    phi, E four vector into px, py, pz, E. It is hardcoded
    to act on several jet 4 vectors which are used in the jet 
    calibration project. 

    Arguments:
    jets (dict of arrays) - the dictionary with jet data

    Returns:
    (dict of arrays) - dictionary with the converted data
    """

    # Initialize return dict
    rd = {}

    # Hardcode the names and ordering of the relevant branches
    prefixes = ['jet_Pileup_', 'jet_', 'jet_JES_', 'jet_true_']
    pt_names = ['jet_PileupPt', 'jet_pt', 'jet_JESPt', 'jet_true_pt']
    eta_names = ['jet_PileupEta', 'jet_eta', 'jet_JESEta', 'jet_true_eta']
    phi_names = ['jet_PileupPhi', 'jet_phi', 'jet_JESPhi', 'jet_true_phi']
    en_names = ['jet_PileupE', 'jet_E', 'jet_JESE', 'jet_true_e']

    # Loop through jet 4 vectors
    for pf, pt, eta, phi, en in zip(prefixes, pt_names, eta_names, phi_names, en_names):

        # Calculate px, py, pz
        px = jets[pt] * np.cos(jets[phi])
        py = jets[pt] * np.sin(jets[phi])
        pz = jets[pt] * np.sinh(jets[eta])

        # Add to return dict
        rd[pf+'px'] = px
        rd[pf+'py'] = py
        rd[pf+'pz'] = pz
        rd[pf+'pt'] = jets[pt]
        rd[pf+'E'] = jets[en]

    return rd

# Some code to test preprocessing
if __name__ == '__main__':

    # Test code won't work without modifying code a bit.

    # Setup some fake jet data
    test_pt = ak.Array([[2e2, 3e3, 4e4, 5e5, 2e5, 8e4],[9e1, 8e2, 7e3]])
    test_e = 2 * test_pt
    test_eta = ak.Array([[1, 0.95, 0.98, 1.01, 1.05, 1.03],[-1, -0.98, -1.02]])
    test_phi = ak.Array([[-3.14, 3.14, -3.1, -3.12, -3.11, 3.08],[3.1, 2.8, -3.14]])
    print("Raw data")
    print(test_pt)
    print(test_eta)
    print(test_phi)

    # Sort by pt
    pt_zero = ak.pad_none(test_pt, 200, axis=1, clip=True)
    pt_zero = ak.to_numpy(ak.fill_none(pt_zero, 0, axis=None))
    indeces = np.argsort(pt_zero, axis=1)

    # Energy normalization
    jets_dict = {'fjet_clus_pt': test_pt, 'fjet_clus_E': test_e}
    pt_pp, en_pp = energy_norm(jets_dict, indeces)
    print("\nNormed energy")
    print(pt_pp)
    print(en_pp)

    # Angular manipulation
    jets_dict = {'fjet_clus_eta': test_eta, 'fjet_clus_phi': test_phi}
    eta_pp, phi_pp = simple_angular(jets_dict, indeces)
    print("\nPreprocessed angular")
    print(eta_pp)
    print(phi_pp)
