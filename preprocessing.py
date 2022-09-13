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

    args and returns are standard (see above)
    """

    # Initialize preprocess dict
    preprocess = {}

    # Loop through target constituents
    for name in params['t_constit_branches']:

        # Get branch
        branch = jets[name]

        # Zero pad
        temp = ak.pad_none(branch, params['max_constits'], axis=1, clip=True)
        temp = ak.to_numpy(ak.fill_none(temp, 0, axis=None))

        # Set small pT constituents to zero
        temp[zero_indeces] = 0

        # We sort by DECREASING pT, which necessitates complicated indexing
        temp = np.take_along_axis(temp, sort_indeces, axis=1)[:,::-1]

        # Write to batch dict
        preprocess[name] = temp

    # Finally we return dict
    return preprocess

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


def log_norm(jets, name, indeces, max_constits=200, **kwargs):
    """ log_norm - As opposed to energy norm, this preprocessing function will
    return the logs of the normalized, and un-normalized pT or E. It is meant
    to recreate the preprocessing used in the Particle Net paper.

    Arguments:
    jets (dict): Dictionary whose elements are awkard arrays giving the constituent
    pt and energy. Usually a batch of a loop over a .root file using uproot.
    name (str): Either 'fjet_clus_pt', or 'fjet_clus_E'. Function will apply preprocessing
    to this element of the jets dict.
    indeces (array): The indeces which will sort the constituents by INCREASING pt. Sort
    will be reflected to sort by decreasing pt.
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

    # Sort jets by decreasing pt
    # Odd slicing is needed to invert pt ordering (decreasing vs increasing)
    log_cons_sort = np.take_along_axis(log_cons_zero, indeces, axis=1)[:,::-1]
    lognorm_cons_sort = np.take_along_axis(lognorm_cons_zero, indeces, axis=1)[:,::-1]

    # Return results
    return log_cons_sort, lognorm_cons_sort


def simple_angular(jets, indeces, max_constits=200, **kwargs):
    """ simple_angular - This function will perform a simple preprocessing
    on the angular constituent information. It is the same preprocessing used
    in 1902.09914.

    Arguments:
    jets (dict): Dictionary whose elements are awkard arrays giving the constituent
    eta and phi. Usually a batch of a loop over a .root file using uproot.
    indeces (array): The indeces which will sort the constituents by INCREASING pt. Sort
    will be reflected to sort by decreasing pt.
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
    first_eta = eta[ax_index, indeces[:,-1]]
    first_phi = phi[ax_index, indeces[:,-1]]

    # Now center
    eta_center = eta - first_eta[:,np.newaxis]
    phi_center = phi - first_phi[:,np.newaxis]

    # Fix discontinuity in phi at +-pi
    phi_center = np.where(phi_center > np.pi, phi_center - 2*np.pi, phi_center)
    phi_center = np.where(phi_center < -np.pi, phi_center + 2*np.pi, phi_center)

    # 2. Rotate such that 2nd hardest constituent sits on negative phi axis
    # Screen indeces for any jets with 1 or 2 constituents (ask about these)
    second_eta = eta_center[ax_index, indeces[:,-2]]
    second_phi = phi_center[ax_index, indeces[:,-2]]
    angle = np.arctan2(second_phi, second_eta) + np.pi/2
    eta_rot = eta_center * np.cos(angle[:,np.newaxis]) + phi_center * np.sin(angle[:,np.newaxis])
    phi_rot = -eta_center * np.sin(angle[:,np.newaxis]) + phi_center * np.cos(angle[:,np.newaxis])

    # 3. If needed, reflect 3rd hardest constituent into positive eta half-plane
    third_eta = eta_rot[ax_index, indeces[:,-3]]
    parity = np.where(third_eta < 0, -1, 1)
    eta_flip = eta_rot * parity[:,np.newaxis]

    # Ready to zero pad eta/phi and send arrays to numpy
    eta_zero = ak.pad_none(eta_flip, max_constits, axis=1, clip=True)
    eta_zero = ak.to_numpy(ak.fill_none(eta_zero, 0, axis=None))
    phi_zero = ak.pad_none(phi_rot, max_constits, axis=1, clip=True)
    phi_zero = ak.to_numpy(ak.fill_none(phi_zero, 0, axis=None))

    # Sort constituents using indeces passed into function
    eta_sort = np.take_along_axis(eta_zero, indeces, axis=1)[:,::-1]
    phi_sort = np.take_along_axis(phi_zero, indeces, axis=1)[:,::-1]


    # Finished preprocessing. Return results
    return eta_sort, phi_sort



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
