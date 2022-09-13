""" set_builder.py - This program will define a class which generates an
.h5 file ready for training, starting from two sets of intermediates files,
one of which will serve as signal and the other background. It can also
calculate training weights, and perform simple standardizations.

Author: Kevin Greif
Last updated 7/5/22
python3
"""

import glob
import h5py
import numpy as np

import processing_utils as pu


class SetBuilder:
    """ SetBuilder - This class provides methods which build a training
    ready .h5 file out of 2 sets of intermediate .h5 files. All jets will be
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

        # Perform train / test split if desired, build schedule
        if setup_dict['test_name'] != None:

            print("Running signal and background w/ split")
            assert self.run_bkg

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

        elif self.run_bkg:

            print("Running signal and background")

            self.schedule.append({'sig': sig_list, 'bkg': bkg_list, 'test': False})

        else:

            print("Running signal without labels")

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
        hl_branches = ref.attrs.get('hl')
        jet_branches = ref.attrs.get('jet')
        max_constits = ref.attrs.get('max_constits')

        # Loop through process list
        for i, file in enumerate(out_list):

            # Number of jets depends on the value of i
            if i == 0:
                n_jets = self.n_train
            elif i == 1:
                n_jets = self.n_test

            # Constituents
            constit_shape = (n_jets, max_constits)
            for var in constit_branches:
                file.create_dataset(var, constit_shape, dtype='f4')

            # HL variables
            hl_shape = (n_jets,)
            for var in hl_branches:
                file.create_dataset(var, hl_shape, dtype='f4')

            # Jet 4 vector
            for var in jet_branches:
                file.create_dataset(var, hl_shape, dtype='f4')

            # Labels
            if self.run_bkg:
                file.create_dataset('labels', hl_shape, dtype='i4')

            # Attributes
            file.attrs.create("num_jets", n_jets)
            file.attrs.create("num_cons", len(constit_branches))
            file.attrs.create("num_hl", len(hl_branches))
            file.attrs.create("num_jet_features", len(jet_branches))
            file.attrs.create("jet", jet_branches)
            file.attrs.create("constit", constit_branches)
            file.attrs.create("hl", hl_branches)
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
                hl_branches = sig.attrs.get('hl')
                jet_branches = sig.attrs.get('jet')
                unstacked = np.concatenate((constit_branches, hl_branches, jet_branches))

                # Get random seed for our shuffles
                rng_seed = np.random.default_rng()
                rseed = rng_seed.integers(1000)

                # Concatenate, Shuffle, and Write each branch
                for var in unstacked:
                    sig_var = sig[var][...]
                    bkg_var = bkg[var][...]
                    this_var = np.concatenate((sig_var, bkg_var), axis=0)
                    self.branch_shuffle(this_var, seed=rseed)
                    target[var][start_index:stop_index,...] = this_var

                # Build labels branch
                sig_labels = np.ones(num_sig_jets)
                bkg_labels = np.zeros(num_bkg_jets)
                labels = np.concatenate((sig_labels, bkg_labels))

                # Shuffle and write labels branch
                self.branch_shuffle(labels, seed=rseed)
                target['labels'][start_index:stop_index] = labels

                # Increment counters and close files
                start_index = stop_index
                sig.close()
                bkg.close()

            # End file loop
        # End schedule loop

        # Finish by printing summary of how many jets were written to file
        print("We wrote", stop_index, "jets to target file")
        target.attrs.modify("num_jets", stop_index)


    def solo_process(self):
        """ solo_process - Identical to the process function, except for use
        when we are only processing one sample (i.e. the set is not for training
        but only for plotting and inference purposes). No labels will be added
        to the set in this function.

        Here we always assume target file is self.train

        No arguments or returns
        """

        # This function should only be called when not running background
        assert not self.run_bkg

        # Only one element of schedule
        dict = self.schedule[0]

        # Counter to keep track of writing in target file
        start_index = 0

        # Loop through file list
        print("Writing to output file")
        for i, name in enumerate(dict['sig']):
            print("Now processing file:{}".format(name))

            # Open file
            file = h5py.File(name, 'r')

            # Find number of jets
            num_jets = file.attrs.get('num_jets')
            stop_index = start_index + num_jets

            # Extract dataset names from attributes
            constit_branches = file.attrs.get('constit')
            hl_branches = file.attrs.get('hl')
            jet_branches = file.attrs.get('jet')
            unstacked = np.concatenate((constit_branches, hl_branches, jet_branches))

            # Get random seed for our shuffles
            rng_seed = np.random.default_rng()
            rseed = rng_seed.integers(1000)

            # Shuffle, and write each branch
            for var in unstacked:
                this_var = file[var][...]
                self.branch_shuffle(this_var, seed=rseed)
                self.train[var][start_index:stop_index,...] = this_var

            # Increment counters and close file
            start_index = stop_index
            file.close()

        # End file loop

        # Finish by printing summary of how many jets were written to file
        print("We wrote", stop_index, "jets to target file")
        self.train.attrs.modify("num_jets", stop_index)



    def branch_shuffle(self, branch, seed=42):
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
