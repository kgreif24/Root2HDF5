# Root2HDF5

This repository is intended to produce .h5 files from ROOT nTuples, which can then be used to train ML models. It has been used in various jet tagging and calibration projects, therefore the functionality is currently limited to processing jet information. If you're considering using this repository as part of an ML pipeline within ATLAS, you should also look at the [FTAG dumpster](https://training-dataset-dumper.docs.cern.ch/), which provides essentially the same functionality but with the added benefit of dumping directory from derivations or even AODs. In addition the code is better written and documented. However if you're interested in dumping .h5 from either flattened or unflattened nTuples containing jets, then keep reading!

## Processing Overview

This repository uses a somewhat complicated 2-stage processing routine, in order to solve the problem of large data sets which can not be loaded into memory all at once. First, a set of ROOT nTuples are processed into "intermediate" files, which contain a subset of the data in the final form desired for piping data into an ML model. Second, these intermediates are independently shuffled and then combined into final training and testing .h5 files. The first step is handled by the `RootConverter` class, and the second is handled by `SetBuilder`.

### RootConverter

This class uses the `uproot.iterate` function to loop over a set of ROOT nTuples. It has a lot of functionality beyond just dumping jet information, including applying systematic variations and running inference over neural networks for the purposes of deriving weights. However for the purposes of this README, we can just focus on dumping jet information.

The `RootConverter` needs to be provided a python dictionary for configuration. Scripts which create such dictionaries and then run the processing are stored in the `config` sub-directory. Here's a sample config (from `r2h_calib.py`):

https://github.com/kgreif24/Root2HDF5/blob/1d9b3dffcfda0bfcf024e9e81e504008ff3e62ee/config/r2h_calib.py#L13-L59

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

This class builds final training and testing .h5 files from a set of intermediates. The SetBuilder can process both a signal and background sample in the case the .h5 will be used to train a classifier, or just a single sample in the case the .h5 will be used to train a regression model. Like the `RootConverter`, this class requires a config python dictionary, several examples of which are in the `config` directory. Here is one 

