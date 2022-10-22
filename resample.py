""" resample.py - This script will resample any .h5 file using the 'weights'
branch. It will produce a sample with the desired jet pT spectrum, with the
number of jets given by a keyword argument

Author: Kevin Greif
python3
Last updated 10/22/22
"""

import h5py
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--in', type=str, required=True,
                    help='The input file name')
parser.add_argument('--branch', type=str, required=True,
                    help='The name of the jet pt branch')
parser.add_argument('--n_jets', type=int, required=True,
                    help='The number of jets to have in the final set')
parser.add_argument('--out', type=str, required=True,
                    help='The output file name')
parser.parser_args()


# Set step size for reading / writing between h5 files
stepsize = 1000000

# Pull jet pt from input file
input = h5py.File(args.in, 'r')
jet_pt = input[args.branch][:]
weights = input['weights'][:]

# Get indeces of jets to keep
rng = np.random.default_rng()
keep = rng.choice(numpy.arange(len(weights)), size=args.n_jets, p=weights)

# Open output file
output = h5py.File(args.out, 'w')

# Loop through keys in input file
for key in input.keys():

    # Get input branch details and create output branch
    in_shape = input[key].shape
    in_type = input[key].dtype
    out_shape = (args.n_jets,) + in_shape[1:]
    output.create_dataset(key, out_shape, dtype=in_type)

    # Initialize counters
    in_start = 0
    out_start = 0

    # Loop over key in input file
    while start < args.n_jets:

        # Pull this batch and keep indeces within
        in_stop = in_start + stepsize
        batch = input[key][start:stop,...]
        batch_idx = keep[(start <= keep)*(keep < stop)]

        # Resample jets
        batch = batch[batch_idx,...]

        # Write jets
        out_stop = out_start + len(batch_idx)
        output[key][out_start:out_start,...] = batch

        # Advance counters
        in_start = in_stop
        out_start = out_stop

# Finish by closing files
input.close()
output.close()
