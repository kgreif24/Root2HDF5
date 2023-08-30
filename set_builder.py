""" set_builder.py - This program will define a class which generates an
.h5 file ready for training, starting from intermediates files. It can also
calculate training weights, and perform simple standardizations (to be added!).

Author: Kevin Greif
Last updated 1/30/23
python3
"""

import sys, os
import glob
import h5py
import numpy as np

import processing_utils as pu


class SetBuilder:
    """ SetBuilder - This class provides methods which build a training
    ready .h5 file out of intermediate .h5 files. All jets will be
    shuffled by default, instructions for calculating training weights
    and standardizations are set in params dictionary passed into init func.
    """

    def __init__(self, setup_dict):
        """ __init__ - The init function for this class accepts a dictionary as
        argument which contains all information needed to make the dataset.
        See the main program below for an example dictionary.

        The function itself uses glob to form lists of all intermediate files
        which will be included in the final sets.

        It also defines a train / test split, if any is desired

        Arguents:
        setup_dict (dict): The dictionary of set parameters

        Returns:
        None
        """

        # Make params dict a class instance variable
        self.params = setup_dict

        # Build schedule of intermediate files to process
        if setup_dict['background'] != None:

            print("Running with background")
            sig_list = sorted(glob.glob(setup_dict['signal'] + '*.h5'))
            bkg_list = sorted(glob.glob(setup_dict['background'] + '*.h5'))
            assert len(bkg_list) == len(sig_list) # Need to have equal numbers of intermediates
            self.schedule = list(zip(sig_list, bkg_list))
            self.run_bkg = True

        else:

            print("Running with only signal")
            sig_list = sorted(glob.glob(setup_dict['signal'] + '*.h5'))
            self.schedule = [(sig,) for sig in sig_list]
            self.run_bkg = False

        # Apply cut on file lists if needed
        if setup_dict['n_files'] != -1:
            cut = setup_dict['n_files']
            self.schedule = self.schedule[:cut]

        # Decide whether we are running train / test split
        if setup_dict['test_name'] != None:
            print("Performing train / test split with modulo {}".format(setup_dict['train_test_mod']))
            self.split = True
        else:
            print("Only building training set")
            self.split = False

        # Count number of jets that will end up in target files
        self.n_train = 0
        self.n_test = 0
        for sched in self.schedule:
            # Add to both counters if we are doing split
            if self.split:
                self.n_train += pu.find_modulo_len(sched, setup_dict['train_test_mod'])
                self.n_test += pu.find_modulo_len(sched, setup_dict['train_test_mod'], is_multiple=True)
            # Otherwise just add entire intermediate to training counter
            else:
                self.n_train += pu.find_h5_len(sched)

        print("Generating files with the following numbers of jets:")
        print("Training set:", self.n_train)
        print("Testing set:", self.n_test)


    def build_files(self):
        """ build_files - This function creates the .h5 files that constitute
        the final set. Will create either 1 or 2 files depending on whether
        we are building a testing set.

        No argument and returns
        """

        print("Building output files")

        # Initialize target list
        target_list = []

        # Build train file
        self.train = h5py.File(self.params['train_name'], 'w')
        target_list.append(self.train)

        # If needed build test file
        if self.split:
            self.test = h5py.File(self.params['test_name'], 'w')
            target_list.append(self.test)

        # Open reference file
        # Currently using first signal intermediate as reference
        ref = h5py.File(self.schedule[0][0], 'r')
        self.constit_branches = ref.attrs.get('constit')
        if 'taste' in '\t'.join(self.constit_branches):
            self.constit_branches = [br for br in self.constit_branches if not 'taste' in br]
            self.taste_branches = ['fjet_clus_taste']
        else:
            self.taste_branches = []
        self.jet_branches = ref.attrs.get('jet')
        self.hl_branches = ref.attrs.get('hl')
        self.event_branches = ref.attrs.get('event')
        self.image_branches = ref.attrs.get('image')
        self.weight_branches = ref.attrs.get('weights')
        self.max_constits = ref.attrs.get('max_constits')

        # Loop through target list
        for i, file in enumerate(target_list):

            # Set number of jets depending on target
            if i == 0:
                n_jets = self.n_train
            elif i == 1:
                n_jets = self.n_test

            # Find shapes of data sets
            if self.params['stack_constits']:
                constit_shape = (n_jets, self.max_constits, len(self.constit_branches))
            else:
                constit_shape = (n_jets, self.max_constits)

            taste_shape = (n_jets, self.max_constits)

            if self.params['stack_jets']:
                jet_shape = (n_jets, len(self.params['jet_fields']))
            else:
                jet_shape = (n_jets,)

            event_shape = (n_jets,)

            # Hardcode image shape, this should not change given what is stored are 
            # bin coordinates in 64x64 grid
            image_shape = (n_jets, self.max_constits, 2)

            # Constituents
            if self.params['stack_constits']:
                file.create_dataset('constit', constit_shape, dtype='f4')
            else:
                for var in self.constit_branches:
                    file.create_dataset(var, constit_shape, dtype='f4')

            # Taste information
            if len(self.taste_branches) != 0:
                file.create_dataset('fjet_clus_taste', taste_shape, dtype='i4')

            # Jet information
            if self.params['stack_jets']:
                for key in self.params['jet_keys']:
                    file.create_dataset(key, jet_shape, dtype='f4')
            else:
                for var in self.jet_branches:
                    file.create_dataset(var, jet_shape, dtype='f4')

            # HL Variable information
            if self.params['stack_hlvars']:
                nvars = len(self.hl_branches)
                hl_shape = (n_jets, nvars)
                file.create_dataset('hl', hl_shape, dtype='f4')
            else:
                for var in self.hl_branches:
                    file.create_dataset(var, jet_shape, dtype='f4')

            # Event information
            for var in self.event_branches:
                file.create_dataset(var, event_shape, dtype='f4')

            # Image information
            for var in self.image_branches:
                file.create_dataset(var, image_shape, dtype='i4')

            # Weight information
            for var in self.weight_branches:
                weight_shape = (n_jets, ref[var].shape[1])
                file.create_dataset(var, weight_shape, dtype='f4')

            # Labels
            file.create_dataset('labels', event_shape, dtype='i4')

            # Attributes
            file.attrs.create("num_jets", n_jets)
            file.attrs.create("num_cons", len(self.constit_branches))
            file.attrs.create("num_jet_features", len(self.jet_branches))
            file.attrs.create("jet", self.jet_branches)
            file.attrs.create("hl", self.hl_branches)
            file.attrs.create("jet_fields", self.params['jet_fields'])
            file.attrs.create("jet_keys", self.params['jet_keys'])
            file.attrs.create("constit", self.constit_branches)
            file.attrs.create("taste", self.taste_branches)
            file.attrs.create("event", self.event_branches)
            file.attrs.create("image", self.image_branches)
            file.attrs.create("weights", self.weight_branches)
            file.attrs.create("max_constits", self.max_constits)


    def process(self, rseed=None):
        """ process - This function loads the data from the input intermediates
        and places them into the output .h5 files. Importantly applies uniform
        shuffling across all branches once all data has been written to
        output file.

        Arguments:
            rseed - Random seed to use for shuffling, if not passed use random
        """

        # Currently this function will only work with a bkg set
        assert(self.run_bkg)

        # Indeces for keeping track of writing in the files
        train_start_index = 0
        if self.split:
            test_start_index = 0

        # Loop through schedule
        for f1, f2 in self.schedule:

            print("\nNow processing files: \n{}\n{}".format(f1, f2))

            # Open files
            sig = h5py.File(f1, 'r')
            bkg = h5py.File(f2, 'r')

            # Find which branches are not stacked
            unstacked = [self.event_branches, self.image_branches, self.taste_branches, self.weight_branches]
            if not self.params['stack_constits']:
                unstacked.append(self.constit_branches)
            if not self.params['stack_jets']:
                unstacked.append(self.jet_branches)
            if not self.params['stack_hlvars']:
                unstacked.append(self.hl_branches)
            unstacked = np.concatenate(unstacked)

            # Get random seed for our shuffles
            if rseed == None:
                rng_seed = np.random.default_rng()
                rseed = rng_seed.integers(1000)

            # Produce event number branch and shuffle
            sig_event_numbers = sig['EventInfo_mcEventNumber'][:]
            bkg_event_numbers = bkg['EventInfo_mcEventNumber'][:]
            event_numbers = np.concatenate((sig_event_numbers, bkg_event_numbers), axis=0)
            pu.branch_shuffle(event_numbers, seed=rseed)

            # Evaluate train / test split
            if self.split:
                train_idx = np.asarray(event_numbers % self.params['train_test_mod']).nonzero()[0]
                train_stop_index = train_start_index + len(train_idx)
                test_idx = np.asarray(event_numbers % self.params['train_test_mod'] == 0).nonzero()[0]
                test_stop_index = test_start_index + len(test_idx)
            # Else just put all jets into the training set
            else:
                train_idx = np.arange(0, len(event_numbers), 1)
                train_stop_index = train_start_index + len(train_idx)

            # Concatenate, Shuffle, and Write each branch
            for var in unstacked:
                sig_var = sig[var][:,...]
                bkg_var = bkg[var][:,...]
                this_var = np.concatenate((sig_var, bkg_var), axis=0)
                pu.branch_shuffle(this_var, seed=rseed)
                self.train[var][train_start_index:train_stop_index,...] = this_var[train_idx,...]
                if self.split:
                    self.test[var][test_start_index:test_stop_index,...] = this_var[test_idx,...]

            # Handle stacked constituent branches
            if self.params['stack_constits']:
                stacked_out = pu.stack_branches((sig, bkg), self.constit_branches, seed=rseed)
                self.train['constit'][train_start_index:train_stop_index,...] = stacked_out[train_idx,...]
                if self.split:
                    self.test['constit'][test_start_index:test_stop_index,...] = stacked_out[test_idx,...]

            # Handle stacked jet branches
            if self.params['stack_jets']:
                for key in self.params['jet_keys']:
                    names = [key + fld for fld in self.params['jet_fields']]
                    jet_stacked_out = pu.stack_branches((sig_var, bkg_var), names, seed=rseed)
                    self.train[key][train_start_index:train_stop_index,...] = jet_stacked_out[train_idx,...]
                    if self.split:
                        self.test[key][test_start_index:test_stop_index,...] = jet_stacked_out[test_idx,...]

            # Handle stacked high level variables
            if self.params['stack_hlvars']:
                hl_stacked_out = pu.stack_branches((sig, bkg), self.hl_branches, seed=rseed)
                self.train['hl'][train_start_index:train_stop_index,...] = hl_stacked_out[train_idx,...]
                if self.split:
                    self.test['hl'][test_start_index:test_stop_index,...] = hl_stacked_out[test_idx,...]

            # Build labels branch
            sig_labels = np.ones(sig.attrs.get('num_jets'))
            bkg_labels = np.zeros(bkg.attrs.get('num_jets'))
            labels = np.concatenate((sig_labels, bkg_labels))

            # Shuffle and write labels branch
            pu.branch_shuffle(labels, seed=rseed)
            self.train['labels'][train_start_index:train_stop_index] = labels[train_idx]
            if self.split:
                self.test['labels'][test_start_index:test_stop_index] = labels[test_idx]

            # Increment counters
            train_start_index = train_stop_index
            if self.split:
                test_start_index = test_stop_index
            
            # Close files
            sig.close()
            bkg.close()

            # End file loop

        # Derive training weights
        if self.params['weight_func'] != None:
            print("Calculating training weights")
            pu.calc_weights(self.train, self.params['weight_func'])

        # Calculate standards if needed
        if self.params['standards']:
            print("Calculating standards")

            # Add standards for constituent branches
            if self.params['stack_constits']:
                pu.calc_standards_stack(self.train, 'constit')
            else:
                pu.calc_standards(self.train, self.constit_branches, 'constit')

            # Here add standards for parallel jet quantities (for jet calibrations)
            for key in self.params['jet_keys']:
                if self.params['stack_jets']:
                    pu.calc_standards_stack(self.train, key)
                else:
                    names = [key + fld for fld in self.params['jet_fields']]
                    pu.calc_standards(self.train, names, key)

            # Also add standards for regular jet branches (for hlvar tagger)
            if self.params['stack_hlvars']:
                pu.calc_standards_stack(self.train, 'hl')
            else:
                pu.calc_standards(self.train, self.hl_branches, 'hl')

        # Finish by printing summary of how many jets were written to file
        print("We wrote", train_stop_index, "jets to training file")
        self.train.attrs.modify("num_jets", train_stop_index)
        if self.split:
            print("We wrote", test_stop_index, "jets to testing file")
            self.test.attrs.modify("num_jets", test_stop_index)


    def solo_process(self, rseed=None):
        """ solo_process - Identical to the process function, except for use
        when we are only processing signal (i.e. the set is not for training
        but only for plotting and inference purposes). No labels will be added
        to the set in this function.

        Arguments:
            rseed (int): Random seed for shuffling, if None then a random seed

        No returns
        """

        raise NotImplementedError("Haven't made solo processing work with event number splits!")

        # This function should only be called when not running background
        assert not self.run_bkg

        # Loop through schedule
        for i, d in enumerate(self.schedule):

            # Index for keeping track of writing in the file
            start_index = 0

            # Set target based on if we are running training or testing
            if d['test']:
                print("Writing to testing set")
                target = self.test
            else:
                print("Writing to training set")
                target = self.train

            # Loop through file list
            iterable = d['sig']
            for i, name in enumerate(iterable):
                print("Now processing file:\nSig:{}".format(name))

                # Open file
                file = h5py.File(name, 'r')

                # Find number of jets
                num_jets = file.attrs.get('num_jets')
                stop_index = start_index + num_jets

                # Find which branches are not stacked
                unstacked = [self.event_branches, self.image_branches, self.taste_branches, self.weight_branches]
                if not self.params['stack_constits']:
                    unstacked.append(self.constit_branches)
                if not self.params['stack_jets']:
                    unstacked.append(self.jet_branches)
                if not self.params['stack_hlvars']:
                    unstacked.append(self.hl_branches)
                unstacked = np.concatenate(unstacked)

                # Get random seed for our shuffles
                if rseed == None:
                    rng_seed = np.random.default_rng()
                    rseed = rng_seed.integers(1000)

                # Shuffle, and write each branch
                for var in unstacked:
                    this_var = file[var][...]
                    pu.branch_shuffle(this_var, seed=rseed)
                    target[var][start_index:stop_index,...] = this_var

                # Handle stacked constituent branches
                if self.params['stack_constits']:
                    stacked_out = pu.stack_branches((file,), self.constit_branches, seed=rseed)
                    target['constit'][start_index:stop_index,...] = stacked_out

                # Handle stacked jet branches
                if self.params['stack_jets']:
                    for key in self.params['jet_keys']:
                        names = [key + fld for fld in self.params['jet_fields']]
                        jet_stacked_out = pu.stack_branches((file,), names, seed=rseed)
                        target[key][start_index:stop_index,...] = jet_stacked_out

                # Handle stacked high level variables
                if self.params['stack_hlvars']:
                    hl_stacked_out = pu.stack_branches((file,), self.hl_branches, seed=rseed)
                    target['hl'][start_index:stop_index,...] = hl_stacked_out

                # Add dummy labels, as given in config
                target['labels'][start_index:stop_index] = self.params['dummy_label'] * np.ones((num_jets,))

                # Increment counters and close file
                start_index = stop_index
                file.close()

            # Update num_jets attribute
            print("We wrote", stop_index, "jets to target file")
            target.attrs.modify("num_jets", stop_index)

            # End file loop

            # Derive training weights
            if self.params['weight_func'] != None and not d['test']:
                print("Calculating training weights")
                pu.calc_weights_solo(target, self.params['weight_func'])

            # Calculate standards if needed
            if self.params['standards']:
                print("Calculating standards")

                if self.params['stack_constits']:
                    pu.calc_standards_stack(target, 'constit')
                else:
                    pu.calc_standards(target, self.constit_branches, 'constit')

                for key in self.params['jet_keys']:
                    if self.params['stack_jets']:
                        pu.calc_standards_stack(target, key)
                    else:
                        names = [key + fld for fld in self.params['jet_fields']]
                        pu.calc_standards(target, names, key)

                # Also add standards for regular jet branches (for hlvar tagger)
                if self.params['stack_hlvars']:
                    pu.calc_standards_stack(target, 'hl')
                else:
                    pu.calc_standards(target, self.hl_branches, 'hl')

        # End schedule loop



    def run(self, **kwargs):
        """ run - The main function for the SetBuilder class. It just calls
        all of the proper functions in order.

        Weight calculations and standardization is currently not implemented.
        We won't need this for awhile anyhow :)

        No arguments or returns
        """

        self.build_files()

        # Run appropriate process function depending on if we are running
        # background
        if self.run_bkg:
            self.process(**kwargs)
        else:
            self.solo_process(**kwargs)

        # Close target files
        self.train.close()
        if self.params['test_name'] != None:
            self.test.close()



if __name__ == '__main__':

    build_dict = {
        'signal': './dataloc/transferLearning/int_zprime/',
        'background': './dataloc/transferLearning/int_dijet/',
        'test_name': None,
        'train_name': './dataloc/transferLearning/delphes_zprime_dijet.h5',
        'test_frac': 0
    }

    sb = SetBuilder(build_dict)
    sb.run()
