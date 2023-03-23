""" split.py - This program will split a given .h5 file along the event axis.
It will generate a number of child .h5 files given a maximum number of events
that can be in each child, and then write the data from the mother file into
each of the children.

Use cases for this script are breaking up large .h5 files into more manageable
sized chunks.

Author: Kevin Greif
Last updated 3/23/2023
python3
"""


import argparse
import h5py
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--infile', type=str, required=True,
                    help='The input file name')
parser.add_argument('--max_events', type=int, required=True,
                    help='The maximum number of events to have in each child .h5 file')
parser.add_argument('--outfile', type=str, required=True,
                    help='The output file name stem')
args = parser.parse_args()


# Load mother .h5 file
f = h5py.File(args.infile, 'r')

# Find the number of child .h5 files we need to generate
total_events = f.attrs['num_jets']
n_children = int(np.ceil(total_events / args.max_events))
remainder = total_events % args.max_events

# Indeces to keep track of reading location in mother
start = 0

# Loop to generate children
print("Creating output files at {}".format(args.outfile))
for i in range(n_children):

    # Find number of events to put in this child
    n_events = args.max_events if i != (n_children-1) else remainder
    stop = start + n_events
    print("Writing events {} to {} to child".format(start, stop))

    # Open child and create datasets
    child = h5py.File("{}_{}.h5".format(args.outfile, i), 'w')
    for key in f.keys():
        in_shape = f[key].shape
        in_type = f[key].dtype
        out_shape = (n_events,) + in_shape[1:]
        child.create_dataset(key, out_shape, dtype=in_type)

    # Copy attributes dictionary from input file
    for key, attr in f.attrs.items():
        child.attrs.create(key, attr)

    # Fix num jets attribute
    child.attrs['num_jets'] = n_events

    # Loop through keys in input file
    for key in f.keys():

        # Pull this batch
        batch = f[key][start:stop,...]

        # Write jets
        child[key][:,...] = batch

    # Increment start index
    start = stop

    # Close child
    child.close()

# Close mother
f.close()