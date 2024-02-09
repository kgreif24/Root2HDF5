""" root_converter.py - This file defines the RootConverter class, which is
the class responsible for looping over a list of .root files
and converting them into the h5 file format. A single dataset can
subsequently be built by shuffling the .h5 output files.

Author: Kevin Greif
Last updated 8/23/2023
python3
"""

import json

import numpy as np
import h5py
import uproot
# import ROOT
import awkward as ak
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
            self.raw_file_events = {name: pu.find_cut_len(name, cb, cf) for name in self.files}
        else:
            self.raw_file_events = {name: pu.find_raw_len(
                name,
                self.params['test_name'],
                self.params['flatten']
            ) for name in self.files}

        self.raw_events = sum(list(self.raw_file_events.values()))
        print("We have", self.raw_events, "jets in total")
        print("We wish to keep", self.params['total'], "of these jets")

        # If statement to catch case where we request more jets than we have
        # or where we just want all jets available
        if (self.params['total'] > self.raw_events) or (self.params['total'] == -1):
            self.params['total'] = self.raw_events
            print("We have", self.raw_events, "jets, so keep this many")

        # Find number of jets we actually want to pull from each file
        self.limits = {
            name: round(count * self.params['total'] / self.raw_events) for name, count in self.raw_file_events.items()
        }

        # Adjustment limit numbers to pull all jets from high JZ slices in dijet samples
        if 'dijet' in self.params['source_list']:

            # First find number of jets we have in each JZ slice
            slice_list = np.arange(364702, 364713, 1)
            slice_counts = {}
            for jz in slice_list:
                sc = sum([count for name, count in self.raw_file_events.items() if str(jz) in name])
                slice_counts[str(jz)] = sc

            # I'm just going to hard code which slice to pull all jets from, then evenly
            # spred the rest. I think this should be good enough.
            if self.params['total'] > 1.1e7:
                full_slice = [364702, 364708, 364709, 364710, 364711, 364712]
            else:
                full_slice = [364702, 364710, 364711, 364712]
            full_slice_count = sum([count for jz, count in slice_counts.items() if int(jz) in full_slice])
            print(f"Have {full_slice_count} jets from JZ slices where we want full statistics")
            print(f"These slices are {full_slice}")

            # Subtract full slice count from the total desired jets
            total_even_split = self.params['total'] - full_slice_count
            assert total_even_split > 0

            # Find number of jets to pull from the even split JZ slices
            even_split_count = round(total_even_split / (len(slice_list) - len(full_slice)))

            # Assemble list of number of jets to pull from each slice
            pull_slice_counts = {
                name: count if int(name) in full_slice else even_split_count for name, count in slice_counts.items() 
            }
            print("This is a dijet sample! Here's the number of jets we will pull from each JZ slice")
            formatted_output = json.dumps(pull_slice_counts, indent=4)
            print(formatted_output)

            # Now need to adjust the limits for each individual ROOT file so the slice counts come out correctly
            # Write a quick lambda function
            def find_jz_index(name):
                for jz in slice_list:
                    if str(jz) in name:
                        return int(jz)
                    
            # Use function to find JZ slices of all files
            file_jz_info = {f: [find_jz_index(f), count] for f, count in self.raw_file_events.items()}
                
            # Now adjust limits based on the jz slice and number of jets in each file
            counters = {jz: 0 for jz in slice_list}
            for f, info in file_jz_info.items():
                if counters[info[0]] < pull_slice_counts[str(info[0])]:
                    # If this file will put us over limit
                    if counters[info[0]] + info[1] > pull_slice_counts[str(info[0])]:
                        diff = pull_slice_counts[str(info[0])] - counters[info[0]]
                        counters[info[0]] += diff
                        self.limits[f] = diff
                    # If this file will not put us over limit
                    else:
                        counters[info[0]] += info[1]
                        self.limits[f] = info[1]
                else:
                    # If we are over the number of jets requested from JZ slice, don't use any jets
                    # from this file
                    self.limits[f] = 0

        # Update total as rounding can cause us to be off by just a bit
        if sum(list(self.limits.values())) != self.params['total']:
            self.params['total'] = sum(list(self.limits.values()))
            print("Due to rounding we will instead keep", self.params['total'], "jets")

        # Load cluster systematics map, if needed.
        if self.params['syst_func'] != None:
            self.syst_map = ROOT.TFile(self.params['syst_loc'], 'read')

        # Process weights dictionary
        self.weight_names = []
        self.weight_shapes = []
        if self.params['weight_branches'] != None:
            self.weight_names = [nm['name'] for nm in self.params['weight_branches']]
            self.weight_shapes = [nm['shape'] for nm in self.params['weight_branches']]

        # Compile lists of branches for use in varying points in production
        # Non constituent branches from source
        self.s_non_constit_branches = (
            self.params['s_jet_branches'] + self.params['event_branches'] + self.params['hlvars']
            + self.weight_names
        )
        # All branches needed after making cuts
        self.keep_branches = (
            self.params['s_jet_branches'] + self.params['event_branches'] 
            + self.params['s_constit_branches'] + self.params['hlvars']
            + self.weight_names
        )
        # All branches that need to be pulled from source
        self.source_branches = self.keep_branches + self.params['cut_branches']
        # Jet shaped branches to target
        self.t_jetshape_branches = (
            self.params['t_jet_branches'] + self.params['event_branches']
            + self.params['hlvars']
        )
        # Non constituent branches to target
        self.t_non_constit_branches = (
            self.t_jetshape_branches + self.params['images_branch'] + self.weight_names
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
            print("Making file", filename)
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
                if br == 'EventInfo_mcEventNumber':
                    file.create_dataset(br, jet_size, maxshape=jet_size, dtype='i4')
                else:
                    file.create_dataset(br, jet_size, maxshape=jet_size, dtype='f4')

            for br in self.params['images_branch']:
                img_size = (max_size, 200, 2)
                file.create_dataset(br, img_size, maxshape=img_size, dtype='i4')

            for nm, shp in zip(self.weight_names, self.weight_shapes):
                this_weight_shape = (max_size,) + shp
                file.create_dataset(nm, this_weight_shape, maxshape=this_weight_shape, dtype='f4')

            # Set file attributes
            file.attrs.create('num_jets', 0, dtype='i4')
            file.attrs.create('constit', self.params['t_constit_branches'])
            file.attrs.create('jet', self.params['t_jet_branches'])
            file.attrs.create('hl', self.params['hlvars'])
            file.attrs.create('image', self.params['images_branch'])
            file.attrs.create('event', self.params['event_branches'])
            file.attrs.create('weights', self.weight_names)
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
        for ifile in self.files:

            # Open file using uproot
            print("\nNow processing file", ifile)
            events = uproot.open(ifile)

            # Start a counter to keep track of how many events we have written from file
            jets_from_file = 0

            # Break flag
            hit_file_limit = False

            # Use uproot.iterate to loop through files
            for jet_batch in events.iterate(step_size="100 MB",
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
                    cuts = self.params['cut_func'](batch_data, hlvar_check=self.params['hlvar_check'])
                    # If no jets pass cuts, just continue to next batch
                    if np.count_nonzero(cuts) == 0:
                        print("No jets passed cuts in this batch, skipping!")
                        continue
                    batch_data = {kw: batch_data[kw][cuts,...] for kw in self.keep_branches}

                #################### Fix Units ######################

                # If unit multiplier is not one, multiply all dimensionful branches
                if self.params['unit_multiplier'] != 1:
                    for kw, branch in batch_data.items():
                        if any(s in kw for s in ['_pt', '_E', '_m']):
                            batch_data[kw] = branch * self.params['unit_multiplier'] 

                ################## Calculate Indexing ################

                # Get indeces to sort by increasing pt (will be inverted later)
                if self.params['pt_name'] != None:

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

                ################### Constituents ####################

                # Here call preprocessing function, as set in params dict.
                if self.params['constit_func'] != None:
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
                for i, name in enumerate(self.t_non_constit_branches):

                    # Convert to numpy
                    batch_data[name] = ak.to_numpy(batch_data[name])

                    # Include catch to normalize hlvars with dimensions
                    # Assuming hlvars are in units of MeV
                    if name in ['fjet_Split12', 'fjet_Split23', 'fjet_ECF1', 'fjet_Qw']:
                        batch_data[name] /= 1e6
                    elif name in ['fjet_ECF2']:
                        batch_data[name] /= 1e12
                    elif name in ['fjet_ECF3']:
                        batch_data[name] /= 1e18

                    # Also find batch length here
                    if i == 0:
                        batch_length = batch_data[name].shape[0]

                ##################### Write ########################

                # Check if this batch will go over file limit
                if jets_from_file + batch_length > self.limits[ifile]:
                    print("Have written", jets_from_file, "jets")
                    print("We have", batch_length, "jets in this batch")
                    print("This puts us over limit of", self.limits[ifile], "jets")
                    batch_length = self.limits[ifile] - jets_from_file
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

            for branch, shape in zip(self.weight_names, self.weight_shapes):
                this_weight_shape = (self.start_index[file_num],) + shape
                targ_file[branch].resize(this_weight_shape)


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
