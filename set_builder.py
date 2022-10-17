""" set_builder.py - This program will define a class which generates an
.h5 file ready for training, starting from intermediates files. It can also
calculate training weights, and perform simple standardizations (to be added!).

Author: Kevin Greif
Last updated 9/26/22
python3
"""

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

        ## Make params dict a class instance variable
        self.params = setup_dict

        ## Find lists of intermediate files, add them to processing schedule
        self.schedule = []

        # Signal sample
        sig_list = glob.glob(setup_dict['signal'] + '*.h5')

        # Background sample
        bkg_dir = setup_dict['background']
        if bkg_dir != None:

            bkg_list = glob.glob(bkg_dir + '*.h5')
            self.run_bkg = True

            # Ensure sig and bkg lists have the same lengths
            assert(len(sig_list) == len(bkg_list))

        else:
            self.run_bkg = False

        # If statements for covering 4 possible configurations for .h5 files
        # If we are making train / test split and including background
        if (setup_dict['test_name'] != None) and (self.run_bkg):

            print("Running with train / test split and background")

            # Calculate split
            frac = setup_dict['test_frac']
            split = int(np.around(frac * len(sig_list)))

            # Perform splits
            sig_test = sig_list[:split]
            sig_train = sig_list[split:]
            bkg_test = bkg_list[:split]
            bkg_train = bkg_list[split:]

            # Append to schedule
            self.schedule.append({'sig': sig_test, 'bkg': bkg_test, 'test': True})
            self.schedule.append({'sig': sig_train, 'bkg': bkg_train, 'test': False})

        # If we are making train / test split but not running background
        elif (setup_dict['test_name'] != None) and (not self.run_bkg):

            print("Running with train / test split but no background")

            # Calculate split
            frac = setup_dict['test_frac']
            split = int(np.around(frac * len(sig_list)))

            # Perform split
            sig_test = sig_list[:split]
            sig_train = sig_list[split:]

            # Append to schedule
            self.schedule.append({'sig': sig_test, 'bkg': [], 'test': True})
            self.schedule.append({'sig': sig_train, 'bkg': [], 'test': False})

        # If we are not making train / test split but running background
        elif self.run_bkg:

            print("Running background without train / test split")

            self.schedule.append({'sig': sig_list, 'bkg': bkg_list, 'test': False})

        # If we are not making train / test split or running background
        else:

            print("Running only signal without train / test split")

            self.schedule.append({'sig': sig_list, 'bkg': [], 'test': False})

        ## Find numbers of jets that will end up in files
        self.n_train = 0
        self.n_test = 0
        for d in self.schedule:
            sig_length = sum([pu.find_h5_len(name) for name in d['sig']])
            bkg_length = sum([pu.find_h5_len(name) for name in d['bkg']])
            if d['test']:
                self.n_test += sig_length + bkg_length
            else:
                self.n_train += sig_length + bkg_length

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

        # Initialize output list
        out_list = []

        # Build train file
        self.train = h5py.File(self.params['train_name'], 'w')
        out_list.append(self.train)

        # If needed build test file
        if self.params['test_name'] != None:
            self.test = h5py.File(self.params['test_name'], 'w')
            out_list.append(self.test)

        # Open reference file
        ref = h5py.File(self.schedule[0]['sig'][0], 'r')
        constit_branches = ref.attrs.get('constit')
        jet_branches = ref.attrs.get('jet')
        event_branches = ref.attrs.get('event')
        onehot_branches = ref.attrs.get('onehot')
        max_constits = ref.attrs.get('max_constits')

        # Loop through process list
        for i, file in enumerate(out_list):

            # Number of jets depends on the value of i
            if i == 0:
                n_jets = self.n_train
            elif i == 1:
                n_jets = self.n_test

            # Find shapes of data sets
            if self.params['stack_constits']:
                constit_shape = (n_jets, max_constits, len(constit_branches))
            else:
                constit_shape = (n_jets, max_constits)
            
            if self.params['stack_jets']:
                jet_shape = (n_jets, len(self.params['jet_fields']))
            else:
                jet_shape = (n_jets,)

            event_shape = (n_jets,)

            # For now hardcoded to assume constituent taste is only onehot branch
            onehot_shape = (n_jets, max_constits, 3)

            # Constituents
            if self.params['stack_constits']:
                file.create_dataset('constit', constit_shape, dtype='f4')
            else:
                for var in constit_branches:
                    file.create_dataset(var, constit_shape, dtype='f4')

            # Jet information
            if self.params['stack_jets']:
                for key in self.params['jet_keys']:
                    file.create_dataset(key, jet_shape, dtype='f4')
            else: 
                for var in jet_branches:
                    file.create_dataset(var, jet_shape, dtype='f4')

            # Event information
            for var in event_branches:
                file.create_dataset(var, event_shape, dtype='f4')

            # One hot information
            for var in onehot_branches:
                file.create_dataset(var, onehot_shape, dtype='i4')

            # Labels
            if self.run_bkg:
                file.create_dataset('labels', event_shape, dtype='i4')

            # Attributes
            file.attrs.create("num_jets", n_jets)
            file.attrs.create("num_cons", len(constit_branches))
            file.attrs.create("num_jet_features", len(jet_branches))
            file.attrs.create("jet", jet_branches)
            file.attrs.create("jet_fields", self.params['jet_fields'])
            file.attrs.create("jet_keys", self.params['jet_keys'])
            file.attrs.create("constit", constit_branches)
            file.attrs.create("event", event_branches)
            file.attrs.create("onehot", onehot_branches)
            file.attrs.create("max_constits", max_constits)


    def process(self):
        """ process - This function loads the data from the input intermediates
        and places them into the output .h5 files. Importantly applies uniform
        shuffling across all branches once all data has been written to
        output file.

        No arguments or returns
        """

        # Currently this function will only work with a bkg set
        assert(self.run_bkg)

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

            # Loop through file lists
            iterable = zip(d['sig'], d['bkg'])
            for i, (sig_name, bkg_name) in enumerate(iterable):
                print("Now processing files:\nSig:{}\nBkg:{}".format(sig_name, bkg_name))

                # Open files
                sig = h5py.File(sig_name, 'r')
                bkg = h5py.File(bkg_name, 'r')

                # Find number of jets in sum of signal and background
                num_sig_jets = sig.attrs.get("num_jets")
                num_bkg_jets = bkg.attrs.get("num_jets")
                num_file_jets = num_sig_jets + num_bkg_jets
                stop_index = start_index + num_file_jets

                # Extract dataset names from attributes
                constit_branches = sig.attrs.get('constit')
                jet_branches = sig.attrs.get('jet')
                event_branches = sig.attrs.get('event')
                onehot_branches = sig.attrs.get('onehot')
                if self.params['stack_constits'] and not self.params['stack_jets']:
                    unstacked = np.concatenate((event_branches, jet_branches, onehot_branches))
                elif self.params['stack_constits'] and self.params['stack_jets']:
                    unstacked = np.concatenate((event_branches, onehot_branches))
                else:
                    unstacked = np.concatenate((constit_branches, jet_branches, event_branches, onehot_branches))

                # Get random seed for our shuffles
                rng_seed = np.random.default_rng()
                rseed = rng_seed.integers(1000)

                # Concatenate, Shuffle, and Write each branch
                for var in unstacked:
                    sig_var = sig[var][:num_sig_jets,...]
                    bkg_var = bkg[var][:num_bkg_jets,...]
                    this_var = np.concatenate((sig_var, bkg_var), axis=0)
                    pu.branch_shuffle(this_var, seed=rseed)
                    target[var][start_index:stop_index,...] = this_var

                # Handle stacked constituent branches
                if self.params['stack_constits']:
                    stacked_out = pu.stack_branches((sig, bkg), constit_branches, seed=rseed)
                    target['constit'][start_index:stop_index,...] = stacked_out

                # Handle stacked jet branches
                if self.params['stack_jets']:
                    for key in self.params['jet_keys']:
                        names = [key + fld for fld in self.params['jet_fields']]
                        jet_stacked_out = pu.stack_branches((sig_var, bkg_var), names, seed=rseed)
                        target[key][start_index:stop_index,...] = jet_stacked_out

                # Build labels branch
                sig_labels = np.ones(num_sig_jets)
                bkg_labels = np.zeros(num_bkg_jets)
                labels = np.concatenate((sig_labels, bkg_labels))

                # Shuffle and write labels branch
                pu.branch_shuffle(labels, seed=rseed)
                target['labels'][start_index:stop_index] = labels

                # Increment counters and close files
                start_index = stop_index
                sig.close()
                bkg.close()

            # End file loop

            # Derive training weights
            if self.params['weight_func'] != None and not d['test']:
                print("Calculating training weights")
                pu.calc_weights(target, self.params['weight_func'])

            # Finish by printing summary of how many jets were written to file
            print("We wrote", stop_index, "jets to target file")
            target.attrs.modify("num_jets", stop_index)

        # End schedule loop


    def solo_process(self):
        """ solo_process - Identical to the process function, except for use
        when we are only processing signal (i.e. the set is not for training
        but only for plotting and inference purposes). No labels will be added
        to the set in this function.

        No arguments or returns
        """

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

                # Extract dataset names from attributes
                constit_branches = file.attrs.get('constit')
                jet_branches = file.attrs.get('jet')
                event_branches = file.attrs.get('event')
                onehot_branches = file.attrs.get('onehot')
                if self.params['stack_constits'] and not self.params['stack_jets']:
                    unstacked = np.concatenate((event_branches, jet_branches, onehot_branches))
                elif self.params['stack_constits'] and self.params['stack_jets']:
                    unstacked = np.concatenate((event_branches, onehot_branches))
                else:
                    unstacked = np.concatenate((constit_branches, jet_branches, event_branches, onehot_branches))

                # Get random seed for our shuffles
                rng_seed = np.random.default_rng()
                rseed = rng_seed.integers(1000)

                # Shuffle, and write each branch
                for var in unstacked:
                    this_var = file[var][...]
                    pu.branch_shuffle(this_var, seed=rseed)
                    target[var][start_index:stop_index,...] = this_var

                # Handle stacked constituent branches
                if self.params['stack_constits']:
                    stacked_out = pu.stack_branches((file,), constit_branches, seed=rseed)
                    target['constit'][start_index:stop_index,...] = stacked_out

                # Handle stacked jet branches
                if self.params['stack_jets']:
                    for key in self.params['jet_keys']:
                        names = [key + fld for fld in self.params['jet_fields']]
                        jet_stacked_out = pu.stack_branches((file,), names, seed=rseed)
                        target[key][start_index:stop_index,...] = jet_stacked_out

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

        # End schedule loop



    def run(self):
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
            self.process()
        else:
            self.solo_process()

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
