""" processing_utils.py - This program defines utility functions
that will be used to process data in the root2hdf.py script.

Author: Kevin Greif
python3
Last updated 11/5/21
"""


import numpy as np
import uproot
import h5py
import awkward as ak
# import hep_ml.reweight as reweight


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
    tb = tb[:,:2]

    # Count the number of jets we have left
    if flatten:
        count = ak.sum(ak.count(tb, axis=1))
    else:
        count = ak.sum(ak.count(tb, axis=0))
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

    # Load information needed to make cuts
    events = uproot.open(filename)
    arrays = events.arrays(filter_name=cut_branches)

    # Call cut function on the loaded arrays
    cuts = cut_func(arrays)

    return np.count_nonzero(cuts)


def find_h5_len(filename):
    """ find_h5_len - Take in a path to a .h5 file and returns the length of
    the fjet_pt data set.

    Arguments:
    filename (string) - The path to the file

    Returns
    (int) - The length of the fjet_pt dataset
    """
    f = h5py.File(filename, 'r')
    return f['fjet_pt'].shape[0]


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


def calc_weights(file, weight_func):
    """ calc_weights - This function calculates weights to adjust the pT spectrum of
    the h5 file passed in as arguments. Applies the weight calculation function
    given by weight_func. This function takes pt as an argument and returns weights.

    Arguments:
    file (obj) - The file to calculate weights for, must be writable
    weight_func (function) - The function used to calculate weights. Must take in a
    vector of jet pt and return jet weights.

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

    # Calculate weights for signal
    bkg_weights = weight_func(bkg_pt, sig_pt)

    # Assemble single vector of weights
    weights = np.ones(num_jets, dtype=np.float32)
    weights[bkg_ind] = bkg_weights

    # Create new dataset in file
    weight_shape = (num_jets,)
    weight_data = file.create_dataset("weights", shape=weight_shape, dtype='f4')
    weight_data[:] = weights


def send_data(file_list, target, hl_means, hl_stddevs):
    """ send_data - This function takes in a list of intermediate .h5 files and from
    them constructs train/test files with stacked constituent and hl information. It
    will also transfer over label, pt, and image information.

    Arguments:
    file_list (list) - A list containing strings giving the path of files to add
    target (obj) - h5 file object for the target file. Assumes we have write permissions
    and that the dataset structure of target is exactly the same as the source files.
    hl_means (list) - A list of hl means to apply in standardization
    hl_stddevs (list) - A list of hl std. deviations to apply in standardization. Must
    have the same length as the number of hl vars.

    Returns:
    None
    """

    # Start counter to keep track of write index in target file
    start_index = 0

    # Loop through file list
    for i, file_name in enumerate(file_list):
        print("Now processing file:", file_name)

        # Open file
        file = h5py.File(file_name, 'r')
        num_file_jets = file.attrs.get("num_jets")
        stop_index = start_index + num_file_jets

        # Extract dataset names from attributes
        constit_branches = file.attrs.get('constit')
        hl_branches = file.attrs.get('hl')
        image_branch = file.attrs.get('img')
        pt_branch = file.attrs.get('pt')
        label_branch = file.attrs.get('label')
        unstacked = np.concatenate((image_branch, pt_branch, label_branch))

        # Get random seed for our shuffles
        rng_seed = np.random.default_rng()
        rseed = rng_seed.integers(1000)

        # Constituents
        print("Processing constituents")
        constit_list = []

        for cons in constit_branches:
            this_cons = file[cons][...]
            branch_shuffle(this_cons, seed=rseed)
            constit_list.append(this_cons)

        # Stack all constituents along last axis here
        constits = np.stack(constit_list, axis=-1)
        # And write to target file
        target['constit'][start_index:stop_index,...] = constits
        del constits

        # HL variables
        print("Processing hl vars")
        hlvars_list = []

        for var, mean, stddev in zip(hl_branches, hl_means, hl_stddevs):
            this_var = file[var][:]
            branch_shuffle(this_var, seed=rseed)

            # Catch for annoying ECF functions which have large magnitudes
            if var == 'fjet_ECF3':
                this_var /= 1e10
            elif var == 'fjet_ECF2':
                this_var /= 1e6

            # Standardize variable using information passed in as argument
            stan_var = (this_var - mean) / stddev
            hlvars_list.append(stan_var)

        # Stack all hl variables
        hlvars = np.stack(hlvars_list, axis=-1)
        # And write to target file
        target['hl'][start_index:stop_index,...] = hlvars
        del hlvars

        # Other information, including images, labels, and jet pT
        print("Processing images, labels, and pT")
        for branch in unstacked:
            dataset = file[branch][...]
            branch_shuffle(dataset, seed=rseed)
            target[branch][start_index:stop_index,...] = dataset

        # Increment counters and close file
        start_index = stop_index
        file.close()

    # End by printing summary of how many jets were written to file
    print("We wrote", stop_index, "jets to target file")
    target.attrs.modify("num_jets", stop_index)


def unstacked_send(file_list, target):
    """ unstacked_send - The same function as above, except it skips the stacking
    and hl variable standardization steps. This is for use in generating the
    public facing data set.

    Arguments:
    file_list (list) - List of paths to intermediate files
    target (string) - The target file h5py object

    Returns:
    None
    """

    # Start counter to keep track of write index in target file
    start_index = 0

    # Loop through file list
    for i, file_name in enumerate(file_list):
        print("Now processing file:", file_name)

        # Open file
        file = h5py.File(file_name, 'r')
        num_file_jets = file.attrs.get("num_jets")
        stop_index = start_index + num_file_jets

        # Extract dataset names from attributes
        constit_branches = file.attrs.get('constit')
        hl_branches = file.attrs.get('hl')
        jet_branches = file.attrs.get('pt')
        label_branch = file.attrs.get('label')
        unstacked = np.concatenate((constit_branches, hl_branches, jet_branches, label_branch))

        # Get random seed for our shuffles
        rng_seed = np.random.default_rng()
        rseed = rng_seed.integers(1000)

        # Process data
        for var in unstacked:
            this_var = file[var][...]
            branch_shuffle(this_var, seed=rseed)
            target[var][start_index:stop_index,...] = this_var

        # Increment counters and close file
        start_index = stop_index
        file.close()

    # End by printing summary of how many jets were written to file
    print("We wrote", stop_index, "jets to target file")
    target.attrs.modify("num_jets", stop_index)



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


def calc_standards(file):
    """ calc_standards - This function will calculate the mean and std. deviation of each
    high level variable in a given .h5 file. It will return these standards as two lists.

    Arguments:
    file (string) - The path to the file we will use to calculate the means and standard deviations.

    Returns:
    (list) - The means of each hl var in our file
    (list) - The std. deviations
    """

    # Open file
    f = h5py.File(file, 'r')

    # Pull hl var names
    hl_vars = f.attrs.get('hl')

    # Initialize empty lists
    means_list = []
    stddevs_list = []

    # Loop through hl vars
    for var in hl_vars:

        # Pull data
        data = f[var][:]

        # For variables with large magnitudes (ECFs) divide by a large value to head off
        # overflows in calculating mean and stddev
        if var == 'fjet_ECF3':
            data /= 1e10
        elif var == 'fjet_ECF2':
            data /= 1e6

        # Calculate mean and std. dev
        mean = data.mean()
        stddev = data.std()

        # Append to lists
        means_list.append(mean)
        stddevs_list.append(stddev)

    return means_list, stddevs_list


def common_cuts(batch):
    """ common_cuts - This function will take in a batch of data (almost always as loaded)
    by uproot.iterate and apply the common cuts for Rel22. For when data format does not
    allow uproot to do this for us.

    Arguments:
    batch (obj or dict) - The batch, where branches are accessible by string names

    Returns:
    (array) - A boolean array of len branch.shape[0]. If True, jet passes common cuts
    """

    # Assemble boolean arrays
    cuts = []
    cuts.append(abs(batch['fjet_truthJet_eta']) < 2.0)
    cuts.append(batch['fjet_truthJet_pt'] / 1000. > 350.)
    cuts.append(batch['fjet_numConstituents'] >= 3)
    cuts.append(batch['fjet_m'] / 1000. > 40.)

    # Going to also include cuts on hl var exit codes here
    cuts.append(batch['fjet_Tau1_wta'] != -999)
    cuts.append(batch['fjet_Tau2_wta'] != -999)
    cuts.append(batch['fjet_Tau3_wta'] != -999)
    cuts.append(batch['fjet_Tau4_wta'] != -999)
    cuts.append(batch['fjet_Split12'] != -999)
    cuts.append(batch['fjet_Split23'] != -999)
    cuts.append(batch['fjet_ECF1'] != -999)
    cuts.append(batch['fjet_ECF2'] != -999)
    cuts.append(batch['fjet_ECF3'] != -999)
    cuts.append(batch['fjet_C2'] != -999)
    cuts.append(batch['fjet_D2'] != -999)
    cuts.append(batch['fjet_Qw'] != -999)
    cuts.append(batch['fjet_L2'] != -999)
    cuts.append(batch['fjet_L3'] != -999)
    cuts.append(batch['fjet_ThrustMaj'] != -999)

    # Take and of all cuts
    total_cuts = np.logical_and.reduce(cuts)

    return total_cuts


def signal_cuts(batch):
    """ signal_cuts - Calls the above function to produce the common cuts, but
    also adds a set of signal cuts which should be applied to the Z' sample.

    Arguments:
    batch (obj or dict) - The batch data from which to compute cuts

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
