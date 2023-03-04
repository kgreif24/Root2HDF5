""" root_converter.py - This file defines the RootConverter class, which is
the class responsible for looping over a list of .root files
and converting them into the h5 file format. A single dataset can
subsequently be built by shuffling the .h5 output files.

Author: Kevin Greif
Last updated 1/29/2023
python3
"""

import gc

import numpy as np
import h5py
import uproot
# import ROOT
import awkward as ak
# import tensorflow as tf
# from energyflow.archs import PFN
import processing_utils as pu
import preprocessing as pp

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
        cb = self.params['cut_branches']
        cf = self.params['cut_func']

        if cf != None:
            raw_file_events = [pu.find_cut_len(name, cb, cf) for name in self.files]
        else:
            raw_file_events = [pu.find_raw_len(
                name,
                self.params['test_name'],
                self.params['flatten']
            ) for name in self.files]

        self.raw_file_events = np.array(raw_file_events)
        self.raw_events = np.sum(self.raw_file_events)
        print("We have", self.raw_events, "jets in total")
        print("We wish to keep", self.params['total'], "of these jets")

        # If statement to catch case where we request more jets than we have
        # or where we just want all jets available
        if (self.params['total'] > self.raw_events) or (self.params['total'] == -1):
            self.params['total'] = self.raw_events
            print("We have", self.raw_events, "jets, so keep this many")

        # Find number of jets we actually want to pull from each file
        fractions = self.raw_file_events / self.raw_events
        self.limits = np.around(fractions * self.params['total']).astype(int)

        # Update total as rounding can cause us to be off by just a bit
        if np.sum(self.limits) != self.params['total']:
            self.params['total'] = np.sum(self.limits)
            print("Due to rounding we will instead keep", self.params['total'], "jets")

        # Load cluster systematics map, if needed.
        if self.params['syst_func'] != None:
            self.syst_map = ROOT.TFile(self.params['syst_loc'], 'read')

        # Load NN reweighter, if needed
        if self.params['nn_weights'] != None:
            self.model = tf.keras.models.load_model(self.params['nn_weights']['file'])

        # Compile lists of branches for use in varying points in production
        # Non constituent branches from source
        self.s_non_constit_branches = (
            self.params['s_jet_branches'] + self.params['event_branches']
        )
        # All branches needed after making cuts
        self.keep_branches = (
            self.params['s_jet_branches'] + self.params['event_branches'] 
            + self.params['s_constit_branches']
        )
        # All branches that need to be pulled from source
        self.source_branches = self.keep_branches + self.params['cut_branches']
        # Jet shaped branches to target
        self.t_jetshape_branches = (
            self.params['t_jet_branches'] + self.params['event_branches']
            + self.params['weight_branches']
        )
        # Non constituent branches to target
        self.t_non_constit_branches = (
            self.t_jetshape_branches + self.params['images_branch']
        )
        # All branches to target
        self.target_branches = (
            self.t_non_constit_branches + self.params['t_constit_branches']
        )


    def build_files(self, max_size=4000000):
        """ build_files - Builds the h5 files which we will recieve jet data.

        Arguments:
        max_size (int) - The maximum number of jets that can be written to each
        .h5 file. Default set to 4 million

        Returns:
        None
        """

        # Initialize list to accept file objects
        self.h5files = []

        # Loop through the number of files we want to create
        for file_num in range(self.params['n_targets']):

            # Open file
            filename = self.params['target_dir'] + self.params['name_stem'] + str(file_num) + ".h5"
            file = h5py.File(filename, 'w')

            # Create all datasets in file
            constits_size = (max_size, self.params['max_constits'])
            for br in self.params['t_constit_branches']:
                # If we are dealing with taste info, can make the dataset of type int
                if 'taste' in br:
                    file.create_dataset(br, constits_size, maxshape=constits_size, dtype='i4')
                # Else we should use type float32
                else:
                    file.create_dataset(br, constits_size, maxshape=constits_size, dtype='f4')

            jet_size = (max_size,)
            for br in self.t_jetshape_branches:
                file.create_dataset(br, jet_size, maxshape=jet_size, dtype='f4')

            for br in self.params['images_branch']:
                img_size = (max_size, 200, 2)
                file.create_dataset(br, img_size, maxshape=img_size, dtype='i4')

            # Set file attributes
            file.attrs.create('num_jets', 0, dtype='i4')
            file.attrs.create('constit', self.params['t_constit_branches'])
            file.attrs.create('jet', self.params['t_jet_branches'])
            file.attrs.create('image', self.params['images_branch'])
            file.attrs.create('event', self.params['event_branches'])
            file.attrs.create('weights', self.params['weight_branches'])
            file.attrs.create('max_constits', self.params['max_constits'])

            # Add file to list
            self.h5files.append(file)


    def process(self, **kwargs):
        """ process - Loops through the source file list using uproots iterate
        function. Applies cuts and preprocessing to each batch, then splits
        the batch and writes the fragments to .h5 target files.

        No arguments or returns.
        """

        # Vector of indeces for tracking where to write in file
        self.start_index = np.zeros(self.params['n_targets'], dtype=np.int32)
        # Vector of counters for tracking how many new jets we write
        self.write_events = np.zeros(self.params['n_targets'], dtype=np.int32)

        print("\nStarting processing loop...")

        # Loop through source files
        for num_source, ifile in enumerate(self.files):

            # Open file using uproot
            print("\nNow processing file", ifile)
            events = uproot.open(ifile)

            # Start a counter to keep track of how many events we have written from file
            jets_from_file = 0

            # Break flag
            hit_file_limit = False

            # Use uproot.iterate to loop through files
            for jet_batch in events.iterate(step_size="200 MB",
                                            filter_name=self.source_branches):

                # Initialize batch data dictionary to accept information
                batch_data = {}

                ##################### Flatten #######################

                # Loop over fields in jet batch
                for kw in jet_batch.fields:

                    # Get branch
                    branch = jet_batch[kw]

                    # If we have an event level branch, need to broadcast
                    # array to jet level quantity shape before flattening
                    if kw in self.params['event_branches']:
                        assert(self.params['flatten'])
                        jl_branch = jet_batch[self.params['test_name']]
                        (branch, jl_branch) = ak.broadcast_arrays(
                            branch,
                            jl_branch
                        )

                    # If we are flattening, need to slice to keep only leading
                    # 2 jets (assuming jets are sorted by decreasing pT),
                    # and then flatten the branch with ak.flatten
                    if self.params['flatten']:
                        branch = branch[:,:2,...]
                        branch = ak.flatten(branch, axis=1)

                    # Send branch to batch dictionary
                    batch_data[kw] = branch

                ##################### Make Cuts #####################

                if self.params['cut_func'] != None:
                    cuts = self.params['cut_func'](batch_data)
                    batch_data = {kw: batch_data[kw][cuts,...] for kw in self.keep_branches}

                #################### Fix Units ######################

                # If unit multiplier is not one, multiply all dimensionful branches
                if self.params['unit_multiplier'] != 1:
                    for kw, branch in batch_data.items():
                        if any(s in kw for s in ['_pt', '_E', '_m']):
                            batch_data[kw] = branch * self.params['unit_multiplier'] 

                ################## Calculate Indexing ################

                # Get indeces to sort by increasing pt (will be inverted later)
                pt_name = self.params['pt_name']
                pt = batch_data[pt_name]
                pt_zero = ak.pad_none(pt, self.params['max_constits'], axis=1, clip=True)
                pt_zero = ak.to_numpy(ak.fill_none(pt_zero, 0, axis=1))
                sort_indeces = np.argsort(pt_zero, axis=1)

                # Find indeces of constituents we wish to mask by setting
                # to zero.
                small_pt_indeces = np.asarray(pt_zero < self.params['mask_lim']).nonzero()

                #################### Apply Systs ####################

                if self.params['syst_func'] != None:

                    var_batch = self.params['syst_func'](batch_data,
                                                         self.syst_map,
                                                         **kwargs)
                    batch_data.update(var_batch)

                ##################### NN Weights #####################

                # Calculate weights using NN if needed
                if self.params['nn_weights'] != None:

                    # Run preprocessing for nn evaluation
                    pp_func = self.params['nn_weights']['pp_func']
                    nn_jets = pp_func(batch_data, sort_indeces, small_pt_indeces, self.params)

                    # Stack information along new axis
                    stacked_inputs = np.stack(list(nn_jets.values()), axis=-1)

                    # Run data through network
                    predictions = self.model.predict(stacked_inputs, 
                                                     batch_size=256, 
                                                     verbose=0).flatten()

                    # Store weights
                    weight_key = self.params['nn_weights']['name']
                    batch_data[weight_key] = predictions / (1 - predictions)

                    # Release memory
                    del nn_jets, stacked_inputs, predictions
                    gc.collect()

                ################### Constituents ####################

                # Here call preprocessing function, as set in params dict.
                cons_batch = self.params['constit_func'](batch_data,
                                                         sort_indeces,
                                                         small_pt_indeces,
                                                         self.params)
                batch_data.update(cons_batch)

                ####################### Images ######################

                # Loop through images branch list
                for name in self.params['images_branch']:

                    # Use np.digitize to produce arrays that give indeces of each constituent
                    bins = np.linspace(-2, 2, 65) # 65 array length to drop overflow bins
                    # minus one is to shift indexing such that 0th bin is [-2, ...)
                    binned_eta = np.digitize(batch_data['fjet_clus_eta'], bins) - 1
                    binned_phi = np.digitize(batch_data['fjet_clus_phi'], bins) - 1

                    # Next need to handle overflow bins, set them to 0 or 63 as appropriate
                    binned_eta = np.clip(binned_eta, 0, 63)
                    binned_phi = np.clip(binned_phi, 0, 63)

                    # Stack image data
                    batch_images = np.stack((binned_eta, binned_phi), axis=-1)

                    # Add images to batch data
                    batch_data[name] = batch_images

                ####################### Jet + Event ########################

                # Apply jet level preprocessing if needed
                if self.params['jet_func'] != None:
                    preprocessed_jets = self.params['jet_func'](batch_data)
                    batch_data.update(preprocessed_jets)

                # Loop through jet and event branches
                for name in self.t_non_constit_branches:

                    # Convert to numpy
                    batch_data[name] = ak.to_numpy(batch_data[name])

                # Also find batch length here
                batch_length = pt_zero.shape[0]

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
        for name in self.target_branches:

            # Split branch into n_targets pieces using np.array_split
            branch_splits = np.array_split(batch[name][:length,...], self.params['n_targets'])

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
            img_size = (self.start_index[file_num], 200, 2)

            # Loop through all target branches and resize
            for branch in self.params['t_constit_branches']:
                targ_file[branch].resize(constits_size)

            for branch in self.t_jetshape_branches:
                targ_file[branch].resize(hl_size)

            for branch in self.params['images_branch']:
                targ_file[branch].resize(img_size)


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
