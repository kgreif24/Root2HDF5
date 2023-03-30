# Root2HDF5

This repository is intended to produce .h5 files from ROOT nTuples, which can then be used to train ML models. It has been used in various jet tagging and calibration projects, therefore the functionality is currently limited to processing jet information. If you're considering using this repository as part of an ML pipeline within ATLAS, you should also look at the [FTAG dumpster](https://training-dataset-dumper.docs.cern.ch/), which provides essentially the same functionality but with the added benefit of dumping directory from derivations or even AODs. In addition the code is better written and documented. However if you're interested in dumping .h5 from either flattened or unflattened nTuples containing jets, then keep reading!

## Requirements

Several of the more advanced features of this package, such as applying systematic variations or using neural network reweighters, require having either ROOT or tensorflow installed. By default, these packages are commented out as installing them both together is non-trivial. Beyond these, the code requires uproot, awkward, h5py, numpy, and a few other common packages for processing HEP data.

## Processing Overview

This repository uses a somewhat complicated 2-stage processing routine, in order to solve the problem of shuffling large data sets which can not be loaded into memory all at once. First, a set of ROOT nTuples are processed into "intermediate" files, which contain a subset of the data in the final form desired for piping data into an ML model. Second, these intermediates are independently shuffled and then combined into final training and testing .h5 files. The first step is handled by the `RootConverter` class, and the second is handled by `SetBuilder`.

### RootConverter

This class uses the `uproot.iterate` function to loop over a set of ROOT nTuples. It has a lot of functionality beyond just dumping jet information, including applying systematic variations and running inference over neural networks for the purposes of deriving weights. However for the purposes of this README, we can just focus on dumping jet information.

The `RootConverter` needs to be provided a python dictionary for configuration. Scripts which create such dictionaries and then run the processing are stored in the `config` sub-directory. Here's a sample config (from `r2h_calib.py`):

https://gitlab.cern.ch/kgreif/Root2HDF5/-/blob/master/config/r2h_calib.py#L13-59

A few dictionary items worth highlighting:
- `tree_name`: The name of the TTree containing the jet data
- `source_list`: The name of a text file containing the paths to the ROOT nTuples, with one file per line
- 'flatten': If running over unflattened nTuples (each entry is an event containing a variable number of jets) set to true. Only the leading 2 jets in the events are kept.
- `max_constits`: The maximum number of constituents to store per jet. Since .h5 files require rectangular arrays, jets with fewer than this number of constituents will be zero-padded, and jets with more will be truncated. Jets constituents should always be ordered by decreasing pT in the constituent level preprocessing functions, so only the softest jet constituents are truncated.
- `target_dir`: The path of the directory to store intermediate files.
- `n_targets`: The number of target .h5 files to produce. Should be high enough such that all of the jets in each individual target file fit in memory.
- `constit_func`: The constituent level preprocessing function. Several such functions are provided in `preprocessing.py`.
- `jet_func`: The same for jet level information.
- `test_name`: This is the branch in the ROOT nTuples used to calculate the number of jets in each file, assuming no cuts are being applied.

The remaining important dictionary items are the s (t) constituent (jet) branches. These set the constituent and jet level branches that should be pulled from the nTuples or written to the targets respectively. The source branches should be everything the user needs to pull from the nTuples, and the target branches should be everything the user wants dumped to the targets after applying any preprocessing.

Finally there is also the `event_branches` item. These are event level quantities that can only be processed if running over unflattened nTuples. These quantities are duplicated for jets within the same event.

### SetBuilder

This class builds final training and testing .h5 files from a set of intermediates. The SetBuilder can process both a signal and background sample in the case the .h5 will be used to train a classifier, or just a single sample in the case the .h5 will be used to train a regression model. Like the `RootConverter`, this class requires a config python dictionary, an example of which is in the `config` directory:

https://gitlab.cern.ch/kgreif/Root2HDF5/-/blob/master/config/make_set.py#L14-27

Here's a brief overview of the items:
- `signal`: The path to the directory storing the signal intermediates.
- `background`: The same but for the background intermediates. If the user wishes to only run over a single set of intermediates (in the case of training a regression model), just set this to `None`.
- `n_files`: The number of intermediate files to use in building the final training and testing sets. Set to -1 to use all intermediates. If running both signal and background, the user should ensure that the same number of jets are contained in the intermediates for signal and background, and that the same number of intermediate files exist. If the latter is not true, the code will throw an exception. Its assumed that the user wants training and testing data sets with approximately equal numbers of signal and background jets.
- `train_name`: The name of the training set .h5 file the code will produce.
- `test_name`: The name of the testing set. If the user does not desire to make a train / test split, set this to `None`.
- `test_frac`: The fraction of the **intermediates** that will be used for building the testing set. Note the train / test split is made on the level of intermediate files, not individual jets.
- `stack_constits`: If set to true, the code will stack all constituent branches along the last dimension, and include a `constit` branch in the final .h5 files. Using this feature avoids the need to stack constituents at each batch step when training a model, which can be slow depending on the application.
- `stack_jets`: The same but for the jet branches.
- `jet_fields` and `jet_keys`: In case the user wishes to save multiple versions of jet information (for example MCJES and GSC calibrated jets), set these items appropriately. The `jet_fields` are the data saved for each jet collection, and the `jet_keys` are the names for each individual collection. For example if we are processing the collections `jet_true` and `jet_JES` such that there are branches in the intermediate files like `jet_true_px` and `jet_JES_py`, the user should set `jet_fields` to `['_px', '_py', ...]` and `jet_keys` to `['jet_true', 'jet_JES']`. 
- `weight_func`: If the user wishes to calculate a set of weights which require the entire dataset to be together all at once, this can be done by specifying the weight function here. The application this is meant for is calculating a set of weights to match the pT spectrum in the background to the pT spectrum in the signal. All jets need to be together to access the full pT spectrum. Set to `None` to disable.

## Running the code

1. `RootConverter` step: First copy one of the example `r2h` scripts from the `config` directory into the parent directory. Then, make a .txt file which contains the paths to the ROOT nTuples you wish to run over. Point the config dictionary at this .txt file, and make any other appropriate modifications. Finally, run the python script. Running over a few million jets should take about an hour, assuming you are not doing anything fancy like applying systematic variations. When the code finishes, you should have a set of intermediate files sitting in the appropriate directory.
2. `SetBuilder` step: Again, copy the `make_set` script from the `config` directory to the parent directory. Make appropriate modifications, and then run the python script. This step is faster, usually taking about 20 minutes to process a few million jets. At the end, you should have training and testing .h5 files, depending on the config.
3. Some projects I have used this for require resampling the jets to have particular pT distributions. See below if you need to do this.
4. Train your model!

## Resampling Script

If the user wishes to resample their .h5 to have a flat jet pT distribution, there is also a `resample.py` script which does exactly this. Running this is quite simple. The input / output file name are specified as command line arguments, along with the number of jets the user wishes to have in their final data set. Setting this number appropriately is a somewhat subtle question. Resampling more jets will provide a larger training set, but at some point the resampling will just repeat jets many times, which is not helpful for training. Ideally the number of jets in the resampled dataset should be on about the same order of magnitude as the number of jets in the raw dataset. 

