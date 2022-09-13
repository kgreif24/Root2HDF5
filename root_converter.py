""" root_converter.py - This file defines the RootConverter class, which is
the class responsible for looping over a list of .root dumper output files
and converting them into the h5 file format. A single dataset can
subsequently be built by shuffling the .h5 output files. See README of
data_processing submodule for details.

Author: Kevin Greif
Last updated 7/1/2022
python3
"""

import numpy as np
import h5py
import uproot
import ROOT
import awkward as ak
import processing_utils as pu
import preprocessing as pp
import syst_variations as syst

class RootConverter:
    """ RootConverter - This class' methods handle the conversion of jet
    data from .root files to .h5 files which can be shuffled and combined
    to form jet tagging datasets.
    """

    def __init__(self, setup_dict):
        """ __init__ - The init function for the class takes in a dictionary
        of parameters for the conversion

        setup_dict (dict): See main program below for all of
        the necessary parameters to include
        """

        # First make dictionary an instance variable
        self.params = setup_dict

        # Now calculate derived parameters. Open files list to get names
        listfile = open(self.params['source_list'], "r")
        files = listfile.readlines()
        self.files = [f.rstrip() + self.params['tree_name'] for f in files]

        # Because files come from different pt slices, we need to pull a
        # representative sample of each file for our train/test data sets.
        # Find number of jets we expect to pull from each file
        cb = self.params['cut_branches'] + self.params['hl_branches']
        cf = self.params['cut_func']
        raw_file_events = [pu.find_cut_len(name, cb, cf) for name in self.files]
        self.raw_file_events = np.array(raw_file_events)
        self.raw_events = np.sum(self.raw_file_events)
        print("We have", self.raw_events, "jets in total")
        print("We wish to keep", self.params['total'], "of these jets")

        # If statement to catch case where we request more jets than we have
        if self.params['total'] > self.raw_events:
            self.params['total'] = self.raw_events
            print("Only have", self.raw_events, "jets, so keep this many")

        # Find number of jets we actually want to pull from each file
        fractions = self.raw_file_events / self.raw_events
        self.limits = np.around(fractions * self.params['total']).astype(int)

        # Update total as rounding can cause us to be off by just a bit
        if np.sum(self.limits) != self.params['total']:
            self.params['total'] = np.sum(self.limits)
            print("Due to rounding we will instead keep", self.params['total'], "jets")

        # Lastly load cluster systematics map, if needed.
        # Simply hard code the location of the systematics map on gpatlas
        if self.params['syst_func'] != None:
            syst_loc = '/DFS-L/DATA/whiteson/kgreif/SystTaggingData/cluster_uncert_map_EM.root'
            self.syst_map = ROOT.TFile(syst_loc, 'read')


    def build_files(self, max_size=4000000):
        """ build_files - Builds the h5 files which we will recieve jet data.

        Arguments:
        max_size (int) - The maximum number of jets that can be written to each
        .h5 file. Default set to 4 million

        Returns:
        None
        """

        if self.params['rw_type'] == 'w':
            print("Building .h5 files")
        else:
            print("Loading .h5 files")

        # Initialize list to accept file objects
        self.h5files = []

        # Loop through the number of files we want to create
        for file_num in range(self.params['n_targets']):

            # Open file given rw type
            filename = self.params['target_dir'] + "tt_dijet_samples_" + str(file_num) + ".h5"
            file = h5py.File(filename, self.params['rw_type'])

            # For rw type of 'w', we rebuild all of our files from scratch
            if self.params['rw_type'] == 'w':

                # Create all datasets in file
                constits_size = (max_size, self.params['max_constits'])
                for br in self.params['t_constit_branches']:
                    file.create_dataset(br, constits_size, maxshape=constits_size, dtype='f4')

                hl_size = (max_size,)
                for br in self.params['hl_branches']:
                    file.create_dataset(br, hl_size, maxshape=hl_size, dtype='f4')

                for br in self.params['jet_branches']:
                    file.create_dataset(br, hl_size, maxshape=hl_size, dtype='f4')

                # Set file attributes
                file.attrs.create('num_jets', 0, dtype='i4')
                file.attrs.create('constit', self.params['t_constit_branches'])
                file.attrs.create('hl', self.params['hl_branches'])
                file.attrs.create('jet', self.params['jet_branches'])
                file.attrs.create('max_constits', self.params['max_constits'])

            # Add built or opened file to list
            self.h5files.append(file)


    def process(self, **kwargs):
        """ process - Loops through the source file list using uproots iterate
        function. Applies cuts and preprocessing to each batch, then splits
        the batch and writes the fragments to .h5 target files.

        No arguments or returns.
        """

        # Use values of h5 file attributes to find where to start writing in h5 file
        self.start_index = np.array([targ_file.attrs.get("num_jets") for targ_file in self.h5files])
        # And initialize counter to keep track of how many new jets we write
        self.write_events = np.zeros(self.params['n_targets'])

        print("\nStarting processing loop...")
        print("Initial write positions:", self.start_index)

        # Loop through source files
        for num_source, ifile in enumerate(self.files):

            # Open file using uproot
            print("\nNow processing file", ifile)
            events = uproot.open(ifile)

            # Start a counter to keep track of how many events we have written from file
            jets_from_file = 0

            # Break flag
            hit_file_limit = False

            # Iterate through the files using iterate, filtering out only branches we need
            keep_branches = (self.params['s_constit_branches'] + self.params['hl_branches']
                             + self.params['jet_branches'])
            source_branches = keep_branches + self.params['cut_branches']
            for jet_batch in events.iterate(step_size="200 MB",
                                            filter_name=source_branches):

                # Initialize batch data dictionary to accept information
                batch_data = {}

                ##################### Make Cuts #####################

                cuts = self.params['cut_func'](jet_batch)
                cut_batch = {kw: jet_batch[kw][cuts,...] for kw in keep_branches}

                #################### Apply Systs ####################

                if self.params['syst_func'] != None:

                    var_batch = self.params['syst_func'](cut_batch,
                                                         self.syst_map,
                                                         self.params['s_constit_branches'],
                                                         **kwargs)
                    cut_batch.update(var_batch)

                ################### Constituents ####################

                # Get indeces to sort by increasing pt (will be inverted later)
                pt_name = self.params['pt_name']
                pt = cut_batch[pt_name]
                pt_zero = ak.pad_none(pt, self.params['max_constits'], axis=1, clip=True)
                pt_zero = ak.to_numpy(ak.fill_none(pt_zero, 0, axis=None))
                sort_indeces = np.argsort(pt_zero, axis=1)

                # Find indeces of very small (or zero) pt constituents
                small_pt_indeces = np.asarray(pt_zero < 100).nonzero()

                # Here call preprocessing function, as set in params dict.
                # See class docstring for details
                cons_batch = self.params['constit_func'](cut_batch,
                                                         sort_indeces,
                                                         small_pt_indeces,
                                                         self.params)
                batch_data.update(cons_batch)

                ####################### Jet ########################

                # Simply loop through jet branches, convert to numpy and add
                # to batch_data dict
                for name in self.params['jet_branches']:

                    branch = cut_batch[name]
                    batch_data[name] = ak.to_numpy(branch)

                # Also find batch length here
                batch_length = batch_data[pt_name].shape[0]

                ##################### High Level ###################

                # The same process as jet variables
                for name in self.params['hl_branches']:

                    branch = cut_batch[name]
                    batch_data[name] = ak.to_numpy(branch)

                ##################### Write ########################

                # Check if this batch will go over file limit
                if jets_from_file + batch_length > self.limits[num_source]:
                    print("Have written", jets_from_file, "jets")
                    print("We have", batch_length, "jets in this batch")
                    print("This puts us over limit of", self.limits[num_source], "jets")
                    batch_length = self.limits[num_source] - jets_from_file
                    print("Instead write", batch_length, "jets")

                    # Set break flag
                    hit_file_limit = True

                self.write_branches(batch_data, batch_length)

                #################### Increment ####################

                # Increment jets from file counter
                jets_from_file += batch_length

                # Break if needed
                if hit_file_limit:
                    break

            # End batch loop

        # End source file loop
        # Finally set target attributes
        for targ_file, num_jets in zip(self.h5files, self.start_index):
            targ_file.attrs.modify("num_jets", num_jets)


    def write_branches(self, batch, length):
        """ write_branches - This function will write the branches contained
        in a batch to the target .h5 files.

        Arguments:
        batch (dict): A dictionary containing the batch of data to write
        length (int): The number of jets to write in this batch

        Returns:
        None
        """

        print("Number of jets to write in this batch:", length)

        # Loop over all target branches
        for name, branch in batch.items():

            # Split branch into n_targets pieces using np.array_split
            branch_splits = np.array_split(branch[:length,...], self.params['n_targets'])

            # Find length of each split and end indeces for writing
            split_lengths = [split.shape[0] for split in branch_splits]
            end_index = self.start_index + split_lengths

            # Loop through branch_splits and write to files
            iterable = zip(branch_splits, self.start_index, end_index)
            for targ_num, (write_array, start, stop) in enumerate(iterable):

                # Write branch to correct h5 file with indeces given by start/stop
                self.h5files[targ_num][name][start:stop,...] = write_array

        # Increment write events counter
        self.write_events += split_lengths

        # Advance starting index
        self.start_index = end_index


    def trim_zeros(self):
        """ trim_zeros - Resizes all target .h5 files to the numbers contained
        in self.start_index. This function should only be called after the
        process function, and only when this is the last writing run (i.e.
        no additional jets will be added to the target files).

        No arguments or returns
        """

        print("\nTrimming zeros from datasets")

        # Loop through h5 target files
        for file_num, targ_file in enumerate(self.h5files):
            print("Now processing target file number", str(file_num))

            # Find appropriate size for datasets in this file
            constits_size = (self.start_index[file_num], self.params['max_constits'])
            hl_size = (self.start_index[file_num],)

            # Loop through all target branches and resize
            for branch in self.params['t_constit_branches']:
                targ_file[branch].resize(constits_size)

            hl_size_branches = (self.params['hl_branches']
                                + self.params['jet_branches'])
            for branch in hl_size_branches:
                targ_file[branch].resize(hl_size)


    def run(self, **kwargs):
        """ run - Main function for RootConverter class. It performs the
        processing steps in the standard order.

        No arguments or returns
        """

        self.build_files()
        self.process(**kwargs)
        if self.params['trim']:
            self.trim_zeros()

        # Print a summary
        print("\nAt end of building files:")
        print("Expected", self.params['total'], "jets")
        print("Wrote", int(np.sum(self.write_events)), "jets")
        print("H5 jets written breakdown:", self.write_events)
        print("H5 jets total breakdown:", self.start_index)




if __name__ == '__main__':

    # Define convert_dict which is passed to RootConverter class
    # This particular dict is set up to do raw conversion (no preprocessing)
    convert_dict = {
        'cut_func': pu.signal_cuts,
        'trim': True,
        'source_list': './dat/Zprime_taste.list',
        'tree_name': ':FlatSubstructureJetTree',
        'rw_type': 'w',
        'max_constits': 200,
        'target_dir': './dataloc/intermediates_test/',
        'n_targets': 1,
        'total': 10000,
        'constit_func': pp.raw_preprocess,
        'syst_func': None,
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

    # Run main program
    rc.run()
