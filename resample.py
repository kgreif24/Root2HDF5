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
parser.add_argument('--infile', type=str, required=True,
                    help='The input file name')
parser.add_argument('--n_jets', type=int, required=True,
                    help='The number of jets to have in the final set')
parser.add_argument('--outfile', type=str, required=True,
                    help='The output file name')
args = parser.parse_args()

print("Resampling jets from {}".format(args.infile))

# Set step size for reading from input file
stepsize = 1000000

# Open files
input_file = h5py.File(args.infile, 'r')
output_file = h5py.File(args.outfile, 'w')
n_input_jets = input_file.attrs.get('num_jets')

# Find the number of jets we should sample in each step
n_sample = int(args.n_jets * stepsize / n_input_jets)
print("Will sample {} jets in each step".format(n_sample))

# Initialize random number generator
rng = np.random.default_rng()

# Initialize counters
in_start = 0
out_start = 0

# Create branches in output file
# Unfortunately to do this we need to loop through the branches of the input
# file before the main processing loop
print("Creating output file at {}".format(args.outfile))
for key in input_file.keys():
    in_shape = input_file[key].shape
    in_type = input_file[key].dtype
    out_shape = (args.n_jets,) + in_shape[1:]
    output_file.create_dataset(key, out_shape, dtype=in_type)

# Start reading from input file
# Continue to read until we have amassed the desired number of jets
while out_start < args.n_jets:

    # Pull weights in this batch of input jets
    in_stop = in_start + stepsize
    weights = input_file['weights'][in_start:in_stop]
    weights /= weights.sum()

    # Find indeces for writing to output file
    out_stop = out_start + n_sample
    if out_stop > args.n_jets:
        out_stop = args.n_jets
        # Since this is the last batch, setting n_sample to be the # of jets
        # we still need
        n_sample = out_stop - out_start

    print("\nPulling infile from {} to {}".format(in_start, in_stop))
    print("Writing to outfile from {} to {}".format(out_start, out_stop))
    print("Sampling {} jets".format(n_sample))

    # Decide which jets in this batch to keep
    keep = rng.choice(np.arange(len(weights)), size=n_sample, p=weights, replace=True)

    # Loop through keys in input file
    for key in input_file.keys():

        # Pull this batch
        batch = input_file[key][in_start:in_stop,...]

        # Resample jets
        batch = batch[keep]

        # Write jets
        output_file[key][out_start:out_stop,...] = batch

    # Advance counters
    in_start = in_stop
    out_start = out_stop

# Copy attributes dictionary from input file
for key, attr in input_file.attrs.items():
    output_file.attrs.create(key, attr)

# Fix num jets attribute
output_file.attrs['num_jets'] = args.n_jets

# Finish by closing files
input_file.close()
output_file.close()
