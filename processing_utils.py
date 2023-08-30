""" processing_utils.py - This program defines utility functions
that will be used to process data in the root2hdf.py script.

Author: Kevin Greif
python3
Last updated 10/22/22
"""

from collections.abc import Iterable

import numpy as np
import uproot
import h5py
import awkward as ak
import hep_ml.reweight as reweight


def find_raw_len(filename, test_branch, flatten):
    """ find_len - Take in a path to a .root file and returns the number
    of events in that file.

    Arguments:
    filename (string) - The path to the file (including tree name)
    test_branch (string) - Name of the branch we should use for getting # of jets
    flatten (bool) - If true, must flatten branch before counting jets

    Returns
    (int) - The number of events
    """

    events = uproot.open(filename)

    # Get test branch, and drop 3rd+ jets
    tb = events[test_branch].array()

    # Count the number of jets we have left
    if flatten:
        tb = tb[:,:2]
        count = ak.sum(ak.count(tb, axis=1))
    else:
        count = ak.count(tb)
    return count


def find_cut_len(filename, cut_branches, cut_func):
    """ find_cut_len - Take in a path to a .root file and returns the number of
    events in that file that will pass a set of cuts.

    Arguments:
    filename (string) - The path to the file (including tree name)
    cuts (list) - List of branches needed to calculate cuts
    cut_func (func) - The function which computes our desired cuts

    Returns
    (int) - The number of events in file that will pass cuts
    """

    # Open file
    print("Finding cut length of {}".format(filename))
    events = uproot.open(filename)

    # Initialize empty list for storing arrays of cut decisions
    cuts = []

    # Loop through file using uproot.iterate to save memory
    for batch in events.iterate(filter_name=cut_branches,
                                step_size=110000):

        # Call cut function
        batch_cuts = cut_func(batch)

        # Append to list
        cuts.append(batch_cuts)

    return np.count_nonzero(np.concatenate(cuts))


def find_h5_len(filename):
    """ find_h5_len - Take in a path to a .h5 file and returns the length of
    the data set.

    Arguments:
    filename (string) - The path to the file, or iterable of paths

    Returns
    (int) - The length of the dataset
    """

    if not isinstance(filename, Iterable):
        filename = (filename,)

    counter = 0
    for fname in filename:
        f = h5py.File(fname, 'r')
        counter += f.attrs.get('num_jets')
    return counter


def find_modulo_len(filename, denominator, is_multiple=False):
    """ find_modulo_len - This function uses the branch "EventInfo_mcEventNumber"
    to calculate the number of events in a given .h5 file that have an event
    number that either is or is not a multiple of the given denominator.

    Useful for performing quick train / test split divisions.

    Arguments:
    filename (string) - The path to the file, or iterable of filenames to be added
    denominator (int) - The denominator to use
    is_multiple (bool) - If set to true, return jets that are a multiple of denominator

    Returns:
    (int) - The number of jets in the file that satisfy the condition 
    """

    # If we only pass in a single string, make an interable list
    if isinstance(filename, str):
        filename = (filename,)

    # Count jets in the files
    jet_counter = 0
    for fname in filename:
        f = h5py.File(fname, 'r')
        event_numbers = f['EventInfo_mcEventNumber'][:]
        if is_multiple:
            jet_counter += np.count_nonzero(event_numbers % denominator == 0)
        else:
            jet_counter += np.count_nonzero(event_numbers % denominator)

    return jet_counter


def flat_weights(pt, n_bins=200, **kwargs):
    """ flat_weights - This function will use the hepml reweight function
    to calculate weights that flatten the pT distribution passed in as a numpy
    array pt. This reweighting is done separately for signal/background, so we
    don't need to have both together to do this reweighting.

    Arguments:
    pt (array) - A numpy array containing jet pT
    n_bins (int) - The number of bins to use in the reweighting

    Returns:
    (array) - The array of weights that flattens the pT spectrum.
    """

    # Initialize array of ones
    weights = np.ones(len(pt))

    # Get range of pT spectrum and sample uniform distribution over this range
    ptmin, ptmax = pt.min(), pt.max()
    rng = np.random.default_rng()
    target = rng.uniform(low=ptmin, high=ptmax, size=len(pt))

    # Fit reweighter to uniform distribution
    reweighter = reweight.BinsReweighter(n_bins=n_bins, n_neighs=3)
    reweighter.fit(pt, target=target)

    # Predict new weights
    weights = reweighter.predict_weights(pt)
    weights /= weights.mean()

    return weights


def match_weights(pt, target, n_bins=200):
    """ match_weights - This function will use the hepml reweight function
    to calculate weights which match the pt distribution to the target
    distribution. Usually used to match the bkg pt distribution to the
    signal.

    Arguments:
    pt (array) - Distribution to calculate weights for
    target (array) - Distribution to match
    n_bins (int)

    Returns:
    (array) - vector of weights for pt
    """

    # Fit reweighter to target distribution
    reweighter = reweight.BinsReweighter(n_bins=n_bins)
    reweighter.fit(pt, target=target)

    # Predict new weights
    weights = reweighter.predict_weights(pt)
    weights /= weights.mean()

    return weights


def target_weights(pt, n_bins=200):
    """ target_weights - This function will calculate weights which match
    the given pt distribution to the Pythia dijet distribution. Mostly
    used only for the purpose of correcting for the lack of JZ5.

    Arguments:
    pt (array) - Distribution to calculate weights for
    n_bins (int)

    Returns:
    (array) - vector of weights for pt
    """
    
    # Load target pT from file
    f = h5py.File('./dataloc/train_ln_nominal.h5', 'r')
    targ_pt = f['fjet_pt'][:]
    labels = f['labels'][:]
    bkg_pt = targ_pt[labels == 0]

    # Fit reweighter to target distribution
    reweighter = reweight.BinsReweighter(n_bins=n_bins)
    reweighter.fit(pt, target=bkg_pt)

    # Predict new weights
    weights = reweighter.predict_weights(pt)
    weights /= weights.mean()

    return weights


def calc_weights(file, weight_func):
    """ calc_weights - This function calculates weights to adjust the pT spectrum of
    the h5 file passed in as arguments. Applies the weight calculation function
    given by weight_func. Currently only setup to reweight the background pT
    spectrum.

    Arguments:
    file (obj) - The file to calculate weights for, must be writable
    weight_func (function) - The function used to calculate weights. Must take in
    vectors of background and signal jet pT

    Returns:
    None
    """

    # Pull info from file
    num_jets = file.attrs.get("num_jets")
    pt = file['fjet_pt'][:]
    labels = file['labels'][:]

    # Separate signal and background pt
    indeces = np.arange(0, num_jets, 1)
    sig_ind = indeces[labels == 1]
    bkg_ind = indeces[labels == 0]
    sig_pt = pt[sig_ind]
    bkg_pt = pt[bkg_ind]

    # Calculate weights for background
    bkg_weights = weight_func(bkg_pt, sig_pt)

    # Assemble single vector of weights
    weights = np.ones(num_jets, dtype=np.float32)
    weights[bkg_ind] = bkg_weights

    # Create new dataset in file
    weight_shape = (num_jets,)
    weight_data = file.create_dataset("training_weights", shape=weight_shape, dtype='f4')
    weight_data[:] = weights


def calc_weights_solo(file, weight_func):
    """ calc_weights - This function calculates weights to adjust the pT spectrum of
    the h5 file passed in as arguments. Applies the weight calculation function
    given by weight_func. This solo version of the function is meant for when
    there is no signal and background distinction.

    Arguments:
    file (obj) - The file to calculate weights for, must be writable
    weight_func (function) - The function used to calculate weights. Must take in
    a vector of jet pT

    Returns:
    None
    """

    # Pull info from file
    num_jets = file.attrs.get("num_jets")
    pt = file['jet_true'][:,3]

    # Calculate weights
    weights = weight_func(pt)

    # Create new dataset in file
    weight_shape = (num_jets,)
    weight_data = file.create_dataset("weights", shape=weight_shape, dtype='f4')
    weight_data[:] = weights


def branch_shuffle(branch, seed=42):
    """ branch_shuffle - This shuffle takes in a dataset represented by a numpy array,
    as well as a seed for a random generator. It will then shuffle the branch using numpys
    random shuffle routine.

    Arguments:
    branch (array) - The array to shuffle along the first dimension
    seed (int) - The random seed to use in shuffling

    Returns:
    None - array is shuffled in place
    """

    rng = np.random.default_rng(seed)
    rng.shuffle(branch, axis=0)


def common_cuts(batch, hlvar_check=False):
    """ common_cuts - This function will take in a batch of data (almost always as loaded)
    by uproot.iterate and apply the common cuts for Rel22. For when data format does not
    allow uproot to do this for us.

    Arguments:
    batch (obj or dict) - The batch, where branches are accessible by string names
    hlvar_check (bool) - If true check for -999 exit codes in hlvars

    Returns:
    (array) - A boolean array of len branch.shape[0]. If True, jet passes common cuts
    """

    # Assemble boolean arrays
    cuts = []
    cuts.append(abs(batch['fjet_truthJet_eta']) < 2.0)
    cuts.append(batch['fjet_truthJet_pt'] / 1000. > 350.)
    cuts.append(batch['fjet_m'] / 1000. > 40.)

    # Apply cut on number of constituents, don't use 'fjet_numConstituents' since this
    # isn't updated when we drop consituents when running systematics.
    num_constits = ak.count(batch['fjet_clus_pt'], axis=1)
    cuts.append(num_constits >= 3)

    # Apply hlvar check if requested
    if hlvar_check:
        cuts.append(exit_check(batch))

    # Take and of all cuts
    total_cuts = np.logical_and.reduce(cuts)

    return total_cuts


def common_cuts_test(batch, denominator=22, **kwargs):
    """ common_cuts_test - Calls the above function, but also includes a requirement
    that the jet's event number modulo the given denominator == 0.

    Useful for building testings sets from systematic varied nTuples.

    Arguments:
    As above, plus...
    denominator (int) - The denominator to use in calculating modulus

    Returns:
    As above 
    """

    cuts = []
    cuts.append(common_cuts(batch, **kwargs))
    cuts.append(batch['EventInfo_mcEventNumber'] % denominator == 0)
    total_cuts = np.logical_and.reduce(cuts)

    return total_cuts


def signal_cuts(batch, hlvar_check=False):
    """ signal_cuts - Calls the above function to produce the common cuts, but
    also adds a set of signal cuts which should be applied to the Z' sample.

    Arguments:
    batch (obj or dict) - The batch data from which to compute cuts
    hlvar_check (bool) - If true check for -999 exit codes in hlvars

    Returns:
    (array) - Boolean array representing total cuts
    """

    # Assemble boolean arrays
    cuts = []
    cuts.append(common_cuts(batch))
    cuts.append(abs(batch['fjet_truth_dRmatched_particle_flavor']) == 6)
    cuts.append(abs(batch['fjet_truth_dRmatched_particle_dR']) < 0.75)
    cuts.append(abs(batch['fjet_truthJet_dRmatched_particle_dR_top_W_matched']) < 0.75)
    cuts.append(batch['fjet_ungroomed_truthJet_m'] / 1000. > 140.)
    cuts.append(batch['fjet_truthJet_ungroomedParent_GhostBHadronsFinalCount'] >= 1)
    cuts.append(batch['fjet_ungroomed_truthJet_Split23'] / 1000. > np.exp(3.3-6.98e-4*batch['fjet_ungroomed_truthJet_pt']/1000.))

    # Apply hlvar check if requested
    if hlvar_check:
        cuts.append(exit_check(batch))

    # Take and of all cuts
    total_cuts = np.logical_and.reduce(cuts)

    return total_cuts


def signal_cuts_test(batch, denominator=22, **kwargs):
    """ signal_cuts_test - Calls the above function, but also includes a requirement
    that the jet's event number modulo the given denominator == 0.

    Useful for building testings sets from systematic varied nTuples.

    Arguments:
    As above, plus...
    denominator (int) - The denominator to use in calculating modulus

    Returns:
    As above 
    """

    cuts = []
    cuts.append(signal_cuts(batch, **kwargs))
    cuts.append(batch['EventInfo_mcEventNumber'] % denominator == 0)
    total_cuts = np.logical_and.reduce(cuts)

    return total_cuts


def ttbar_cuts(batch, hlvar_check=False):
    """ ttbar_cuts - Calls the above function to produce the signal cuts,
    but also adds a requirement on the dR between the top and anti-top
    partons in SM ttbar events.

    Arguments:
    batch (obj or dict) - The batch data from which to compute cuts
    hlvar_check (bool) - If true check for -999 exit codes in hlvars

    Returns:
    (array) - Boolean array representing total cuts
    """

    # Assemble boolean arrays
    cuts = []
    cuts.append(signal_cuts(batch))
    cuts.append(batch['EventInfo_ttbar_deltaR'] < 2.0)

    # Apply hlvar check if requested
    if hlvar_check:
        cuts.append(exit_check(batch))

    # Take and of all cuts
    total_cuts = np.logical_and.reduce(cuts)

    return total_cuts


def exit_check(batch):
    """ exit_check - This function will check the standard list of hlvars for exit codes.
    Jets with exit codes will be cut from the dataset. Since all hlvars should be positive,
    just cut anything that is less than zero.

    Arguments:
    batch (obj or dict) - The batch, where branches are accessible by string names

    Returns:
    (array) - A boolean array of len branch.shape[0]. If True, jet passes exit check
    """

    # Assemble boolean arrays
    cuts = []
    hlvar_names = ['fjet_Tau1_wta', 'fjet_Tau2_wta', 'fjet_Tau3_wta', 'fjet_Tau4_wta',
                   'fjet_Split12', 'fjet_Split23', 'fjet_ECF1', 'fjet_ECF2', 'fjet_ECF3',
                   'fjet_C2', 'fjet_D2', 'fjet_Qw', 'fjet_L2', 'fjet_L3', 'fjet_ThrustMaj']
    for name in hlvar_names:
        cuts.append(batch[name] > 0)

    # Take and of all cuts
    total_cuts = np.logical_and.reduce(cuts)

    return total_cuts

def trans_cuts(batch):
    """ trans_cuts - Implements the cuts we will use for both signal and
    background for the Delphes data used in the transfer learning
    project.

    Arugments:
    batch (obj or dict) - The batch of jets for which to compute cuts

    Returns:
    (array) - Boolean array representing total cuts
    """

    # Assemble boolean arrays
    cuts = []
    cuts.append(abs(batch['fjet_eta']) < 2.0)
    cuts.append(batch['fjet_pt'] > 350000)
    cuts.append(batch['fjet_numConstits'] >= 3)
    cuts.append(batch['fjet_m'] > 40)

    # Take and of all cuts
    total_cuts = np.logical_and.reduce(cuts)

    return total_cuts


def calib_cuts(batch):
    """ calib_cuts - Implements the cuts used in the jet calibration project.
    They just require jet eta less than 2.5.

    Arguments:
    batch (obj or dict) - The batch of jets for which to computer cuts

    Returns:
    (array) - Boolean array representing total cuts
    """

    # Assemble boolean arrays
    eta = batch['jet_eta']
    if 'var' in str(ak.type(eta)):
        eta = ak.flatten(eta)
    cuts = []
    cuts.append(abs(eta) < 2.5)

    # Take and of all cuts
    total_cuts = np.logical_and.reduce(cuts)

    return total_cuts


def count_sig(raw_batch, sig=False):
    """ This function will count the number of jets in a raw batch (loaded from
    nTuple before flattening) after applying signal or common cuts.

    Arguments:
    raw_batch (obj or dict) - Usually an object from file.arrays call containing data
    sig (bool) - If true use signal cuts, if false use only common cuts

    Returns:
    (int) - The number of jets in raw_batch that will pass cut
    """

    # Start by flattening batch
    flat_batch = {key: ak.flatten(raw_batch[key], axis=1) for key in raw_batch.fields}

    # Then pass to cutting functions
    cuts = common_cuts(flat_batch)
    if sig:
        sig_cuts = signal_cuts(flat_batch)
        cuts = np.logical_and(cuts, sig_cuts)

    # Return the number of trues in boolean array
    return np.count_nonzero(cuts)


def encode_onehot(y, n_classes):
    """ encode_onehot - This function performs onehot encoding on the branches of
    the batch dict which are given in the "names" array. The number of classes
    are passed in as a vector of the same length as "names".

    Code inspired by the tf.keras.utils.to_categorical function, which I couldn't
    be bothered to get to compile with ROOT!

    Arguments:
    batch (dict) - The dictionary of jet information
    names (array of strings) - The names of the branches to one hot encode
    n_classes (array of ints) - The number of classes in each encoding

    Returns:
    (dict) - A dictionary with the structure {name: onehot_encoded_branch}
    """

    # Convert data to integers and find shape
    y = np.array(y, dtype=np.int32)
    input_shape = y.shape

    # Unravel array
    y = y.ravel()
    n = y.shape[0]

    # Build categorical array
    categorical = np.zeros((n, n_classes), dtype=np.int32)
    categorical[np.arange(n), y] = 1

    # Find output shape and reshape categorical
    output_shape = input_shape + (n_classes,)
    categorical = np.reshape(categorical, output_shape)

    return categorical


def stack_branches(file_list, branches, **kwargs):
    """ stack_branches - This function pull the branches lited in the "branches"
    argument, and stacks them along a new axis. The new axis is added to the end
    of the return array. Shuffling is applied uniformly to each branch

    Arguments:
    file (list of h5 file objects) - The files we wish to pull data from and stack.
    Each must have the branches given.
    branches (list of string) - The branches we wish to stack
    seed (int) - kwargs passed to branch_shuffle function

    Returns:
    (array) - Numpy array of the stacked data
    """

    data_list = []

    for var in branches:

        concat_list = []

        for file in file_list:
            data = file[var][...]
            concat_list.append(data)

        whole_data = np.concatenate(concat_list, axis=0)
        branch_shuffle(whole_data, **kwargs)

        data_list.append(whole_data)

    return np.stack(data_list, axis=-1)


def branch_shuffle(branch, seed=42):
    """ branch_shuffle - This shuffle takes in a dataset represented by a numpy array,
    as well as a seed for a random generator. It will then shuffle the branch using numpys
    random shuffle routine.

    Arguments:
    branch (array) - The array to shuffle along the first dimension
    seed (int) - The random seed to use in shuffling

    Returns:
    None - array is shuffled in place
    """

    rng = np.random.default_rng(seed)
    rng.shuffle(branch, axis=0)


def calc_standards(file, branches, name, max_jets=1000000):
    """ calc_standards - Function takes in a h5py file object, and loops through
    all data branches in the file and calculates the std. dev. and mean
    for each of the branches. These constants can then be used to standardize
    model inputs. Standards are written as attributes of the h5py file.

    This function is for unstacked branches.

    Arguments:
    file (h5py obj) - The h5py file object, must be writeable
    branches (list of str) - List of the branches for which we want standards
    name (str) - The name of the attribute to add to file
    max_jets (int) - The maximum # of jets to consider in calculating avg
    and std. dev.

    Returns:
    None
    """

    # Correct max jets if necessary
    num_jets = file.attrs.get('num_jets')
    if num_jets < max_jets:
        max_jets = num_jets

    # Initialize means and stddevs arrays
    means = np.zeros(len(branches))
    stddevs = np.zeros(len(branches))

    # Loop through branches
    for i, br in enumerate(branches):

        # Pull and flatten branch
        var = np.ravel(file[br][:max_jets, ...])

        # Find means and stddevs
        means[i] = var.mean()
        stddevs[i] = var.std()

    # Write results to file attrs
    file.attrs.create(name + "_means", means)
    file.attrs.create(name + "_stddevs", stddevs)


def calc_standards_stack(file, branch, max_jets=1000000):
    """ calc_standards - Function takes in a h5py file object, and calculates
    the means and stddevs for each dimension of a stacked branch.
    These constants can then be used to standardize
    model inputs. Standards are written as attributes of the h5py file.

    This function is for a stacked branch.

    Attribute will be named after the branch

    Arguments:
    file (h5py obj) - The h5py file object, must be writeable
    branch (str) - Name of the branch
    max_jets (int) - The maximum # of jets to consider in calculating mean
    and std. dev.

    Returns:
    None
    """

    # Correct max jets if necessary
    num_jets = file.attrs.get('num_jets')
    if num_jets < max_jets:
        max_jets = num_jets

    # Get shape of branch
    branch_shape = file[branch].shape

    # Initialize means and stddevs arrays
    means = np.zeros(branch_shape[-1])
    stddevs = np.zeros(branch_shape[-1])

    # Loop through dimensions
    for i in range(branch_shape[-1]):

        # Pull and flatten dimension
        var = np.ravel(file[branch][:max_jets,...,i])

        # Find means and stddevs
        means[i] = var.mean()
        stddevs[i] = var.std()

    # Write results to file attrs
    file.attrs.create(branch + "_means", means)
    file.attrs.create(branch + "_stddevs", stddevs)
