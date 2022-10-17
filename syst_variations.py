""" syst_variations.py - This script defines function which apply systematic
variations to the constituent level inputs. These functions can be passed to
the root converter class to produce data set with applied systematics.

Currently, the variations implemented are the EM scale cluster uncertainties.

For now, we hard code the names of constituent level branches in the jet
batches. (e.g. 'fjet_clus_E'). Or we should, refactor this!

Large amounts of duplicated code currently in this file. Refactoring needed!

Author: Kevin Greif
Last updated 7/5/22
python3
"""

import sys

import awkward as ak
import numpy as np
from tqdm import tqdm

def reco_efficiency(jets, uncert_map, constit_branches):
    """ reco_efficiency - This function applies the cluster reconstruction
    efficiency systematic variation to the constituent level inputs contained
    in the jet batch.

    Arguments:
    jets (dict): The jet batch, almost always as defined in the root converter
    class after cuts have been applied. See root_converter.py for details.
    uncert_map (TFile): The uncertaintiy map file object loaded using PyROOT
    constit_branches (list): The names of the constituent branches to apply
    variation on.

    Returns:
    (dict): A dictionary containing the constituent level quantities with
    applies systematic variation. Keys are identical to those given as input.
    """

    total_counter = 0
    dropped_counter = 0

    # Get cluster scale histogram from uncert map
    cluster_scale = uncert_map.Get('Scale')

    # Convert energy to GeV
    en = jets['fjet_clus_E']

    # In this variation, we build awkward arrays with boolean values so we can
    # apply mask to constituents, dropping the appropriate clusters
    n_jets = len(en)
    n_constits = ak.count(en, axis=1)
    total_constits = ak.sum(n_constits)

    ## Initialize awkward array builder
    builder = ak.ArrayBuilder()

    # Immediately begin list in array builders
    builder.begin_list()

    ## Loop over flattened array, putting list breaks in awkward array based
    # on information in n_constits vector
    en = ak.flatten(en)
    eta = ak.flatten(abs(jets['fjet_clus_eta']))
    taste = ak.flatten(jets['fjet_clus_taste'])
    iterable = zip(en, eta, taste)

    # Counters to manage list breaks
    jet_counter = 0
    constit_counter = 0

    # Initialize rng
    rng = np.random.default_rng()

    # Constituent loop
    for cons_en, cons_eta, cons_taste in iterable:

        ## Start by finding number of constits in a jet, only if this is
        # the first constituent
        if constit_counter == 0:
            jet_constits = n_constits[jet_counter]

        ## If constituent is not neutral (taste == 1), write True
        if cons_taste != 1:
            builder.append(True)

        # Else, we apply systematic variation
        else:

            # Get energy and eta bins
            Ebin = cluster_scale.GetXaxis().FindBin(cons_en)
            ebin = cluster_scale.GetYaxis().FindBin(cons_eta)

            # Correct overflows
            if (Ebin > cluster_scale.GetNbinsX()):
                Ebin = cluster_scale.GetNbinsX()
            elif (Ebin < 1):
                Ebin = 1

            if (ebin > cluster_scale.GetNbinsY()):
                ebin = cluster_scale.GetNbinsY()
            elif (ebin < 1):
                ebin = 1

            # If we have bin content, divide cluster energy by scale
            p = cons_en
            if (cluster_scale.GetBinContent(Ebin, ebin) > 0):
                p = cons_en / cluster_scale.GetBinContent(Ebin, ebin)

            # Now find r, depending on the value of eta
            if (cons_eta < 0.6):
                r = (0.12*np.exp(-0.51*p) + 4.76*np.exp(-0.29*p*p)) / 100
            elif (cons_eta < 1.1):
                r = (0.17*np.exp(-1.31*p) + 4.33*np.exp(-0.23*p*p)) / 100
            elif (cons_eta < 1.4):
                r = (0.17*np.exp(-0.95*p) + 1.14*np.exp(-0.04*p*p)) / 100
            elif (cons_eta < 1.5):
                # This one is a crummy fit, see twiki
                r = (0.15*np.exp(-1.14*p) + 2768.98*np.exp(-4.2*p*p)) / 100
            elif (cons_eta < 1.8):
                r = (0.16*np.exp(-2.77*p) + 0.67*np.exp(-0.11*p*p)) / 100
            elif (cons_eta < 1.9):
                r = (0.16*np.exp(-1.47*p) + 0.86*np.exp(-0.12*p*p)) / 100
            else:
                r = (0.16*np.exp(-1.61*p) + 4.99*np.exp(-0.52*p*p)) / 100

            # Get random number
            flip = rng.uniform()

            # Accept or reject constituent
            if ((flip < r) and (cons_en / 1000 < 2.5)):
                dropped_counter +=1
                builder.append(False)
            else:
                builder.append(True)

            total_counter += 1

        ## Increment consituent counter
        constit_counter += 1

        ## If this is the last constituent in the jet, add array break
        if constit_counter == jet_constits:

            # Add array break
            builder.end_list()
            builder.begin_list()

            # Reset constituent counter
            constit_counter = 0

            # Increment jet counter
            jet_counter += 1

    # End constituent loop
    # Get awkward array from builder
    keep = builder.snapshot()

    # Index all constituent level branches, dropping the required constituents
    var_dict = {kw: jets[kw][keep] for kw in constit_branches}

    print("For this batch we dropped {0:f} percent of constituents".format(dropped_counter * 100 / total_counter))

    return var_dict


def energy_scale(jets, uncert_map, constit_branches, direction='up'):
    """ energy_scale - This function applies the cluster energy scale
    variation to the constituent level inputs contained in the jet batch.

    Arguments:
    jets (dict): The jet batch, almost always as defined in the root converter
    class after cuts have been applied. See root_converter.py for details.
    uncert_map (TFile): The uncertaintiy map file object loaded using PyROOT
    constit_branches (list): The names of the constituent branches to apply
    variation on.
    direction (string): Either 'up' or 'down' to control which direction we
    apply the systematic variation.

    Returns:
    (dict): A dictionary containing the constituent level quantities with
    applies systematic variation.
    """

    # Get cluster scale and mean histogram from uncert map
    cluster_scale = uncert_map.Get('Scale')
    cluster_means = uncert_map.Get('Mean')

    # Pull pt and energy
    pt = jets['fjet_clus_pt']
    en = jets['fjet_clus_E']

    # Loop over jet constituents
    # Instead of building an awkard array with boolean values, we directly
    # build the new pT and energy values for the constituents
    n_jets = len(en)
    n_constits = ak.count(en, axis=1)
    total_constits = ak.sum(n_constits)

    ## Initialize 2 akward array builders for the varied pT and energy arrays
    p_builder = ak.ArrayBuilder()
    E_builder = ak.ArrayBuilder()

    # Immediately begin list in array builders
    p_builder.begin_list()
    E_builder.begin_list()

    # Loop over flattened array, putting list breaks in awkward array based
    # on information in n_constits vector
    en = ak.flatten(en)
    eta = ak.flatten(abs(jets['fjet_clus_eta']))
    pt = ak.flatten(pt)
    taste = ak.flatten(jets['fjet_clus_taste'])
    iterable = zip(en, eta, pt, taste)

    # Counters to manage list breaks
    jet_counter = 0
    constit_counter = 0

    # Constituent loop
    for cons_en, cons_eta, cons_pt, cons_taste in iterable:

        ## Start by finding number of constits in a jet, only if this is
        # the first constituent
        if constit_counter == 0:
            jet_constits = n_constits[jet_counter]

        ## If constituent is not neutral (taste == 1), write nominal values
        if cons_taste != 1:

            # Write nominal, remembering to convert back to MeV
            p_builder.append(cons_pt)
            E_builder.append(cons_en)

        # Else, we apply systematic variation
        else:

            # Get energy and eta bins
            Ebin = cluster_scale.GetXaxis().FindBin(cons_en)
            ebin = cluster_scale.GetYaxis().FindBin(cons_eta)

            # Correct overflows
            if (Ebin > cluster_scale.GetNbinsX()):
                Ebin = cluster_scale.GetNbinsX()
            elif (Ebin < 1):
                Ebin = 1

            if (ebin > cluster_scale.GetNbinsY()):
                ebin = cluster_scale.GetNbinsY()
            elif (ebin < 1):
                ebin = 1

            # If we have bin content, divide cluster energy by scale
            p = cons_en
            if (cluster_scale.GetBinContent(Ebin, ebin) > 0):
                p = cons_en / cluster_scale.GetBinContent(Ebin, ebin)

            # Now get pT bins
            pbin = cluster_means.GetXaxis().FindBin(cons_pt)

            # Correct overflow
            if (pbin > cluster_means.GetNbinsX()):
                pbin = cluster_means.GetNbinsX()
            elif (pbin < 1):
                pbin = 1

            # Find CES
            bc = cluster_means.GetBinContent(pbin, ebin)
            ces = abs(bc - 1)

            # Catch case where we are looking up bin with no entries (???)
            if p > 350 or bc == 0:
                ces = 0.1

            # Apply pT variation
            if direction == 'up':
                ptces = cons_pt * (1 + ces)
            elif direction == 'down':
                ptces = cons_pt * (1 - ces)

            # Calculate new energy
            Eces = ptces * np.cosh(cons_eta)
            # print("\nOld pT: {0:.4f}\tOld en: {1:.4f}".format(cons_pt, cons_en))
            # print("New pT: {0:0.4f}\tNew en: {1:.4f}".format(ptces, Eces))

            # Add new values to array builders
            p_builder.append(ptces)
            E_builder.append(Eces)

        ## Increment consituent counter
        constit_counter += 1

        ## If this is the last constituent in the jet, add array break
        if constit_counter == jet_constits:

            # Add array break
            p_builder.end_list()
            E_builder.end_list()
            p_builder.begin_list()
            E_builder.begin_list()

            # Reset constituent counter
            constit_counter = 0

            # Increment jet counter
            jet_counter += 1

    # End constituent loop

    # Take snapshots of builders
    var_pt = p_builder.snapshot()
    var_en = E_builder.snapshot()

    # Return dictionary with varied pT and energy information
    return {'fjet_clus_pt': var_pt, 'fjet_clus_E': var_en}

def energy_res(jets, uncert_map, constit_branches):
    """ energy_res - This function applies the cluster energy resolution
    variation to the constituent level inputs.

    Arguments:
    jets (dict): The jet batch, almost always as defined in the root converter
    class after cuts have been applied. See root_converter.py for details.
    uncert_map (TFile): The uncertaintiy map file object loaded using PyROOT
    constit_branches (list): The names of the constituent branches to apply
    variation on.
    direction (string): Either 'up' or 'down' to control which direction we
    apply the systematic variation.

    Returns:
    (dict): A dictionary containing the constituent level quantities with
    applies systematic variation.
    """

    # Get cluster energy scale and RMS from uncert map
    cluster_scale = uncert_map.Get('Scale')
    cluster_rms = uncert_map.Get('RMS')

    # Pull pt and energy
    pt = jets['fjet_clus_pt']
    en = jets['fjet_clus_E']

    # Instead of building an awkard array with boolean values, we directly
    # build the new pT and energy values for the constituents
    n_jets = len(en)
    n_constits = ak.count(en, axis=1)
    total_constits = ak.sum(n_constits)

    ## Initialize 2 akward array builders for the varied pT and energy arrays
    p_builder = ak.ArrayBuilder()
    E_builder = ak.ArrayBuilder()

    # Immediately begin list in array builders
    p_builder.begin_list()
    E_builder.begin_list()

    ## Loop over flattened array, putting list breaks in awkward array based
    # on information in n_constits vector
    en = ak.flatten(en)
    eta = ak.flatten(abs(jets['fjet_clus_eta']))
    pt = ak.flatten(pt)
    taste = ak.flatten(jets['fjet_clus_taste'])
    iterable = zip(en, eta, pt, taste)

    # Counters to manage list breaks
    jet_counter = 0
    constit_counter = 0

    # Constituent loop
    for cons_en, cons_eta, cons_pt, cons_taste in iterable:

        ## Start by finding number of constits in a jet, only if this is
        # the first constituent
        if constit_counter == 0:
            jet_constits = n_constits[jet_counter]

        ## If constituent is not neutral (taste == 1), write nominal values
        if cons_taste != 1:

            # Write nominal, converting back to MeV
            p_builder.append(cons_pt)
            E_builder.append(cons_en)

        # Else, we apply systematic variation
        else:

            # Get energy and eta bins
            Ebin = cluster_scale.GetXaxis().FindBin(cons_en)
            ebin = cluster_scale.GetYaxis().FindBin(cons_eta)

            # Correct overflows
            if (Ebin > cluster_scale.GetNbinsX()):
                Ebin = cluster_scale.GetNbinsX()
            elif (Ebin < 1):
                Ebin = 1

            if (ebin > cluster_scale.GetNbinsY()):
                ebin = cluster_scale.GetNbinsY()
            elif (ebin < 1):
                ebin = 1

            # If we have bin content, divide cluster energy by scale
            p = cons_en
            if (cluster_scale.GetBinContent(Ebin, ebin) > 0):
                p = cons_en / cluster_scale.GetBinContent(Ebin, ebin)

            # Now get pT bins
            pbin = cluster_rms.GetXaxis().FindBin(p)

            # Correct overflow
            if (pbin > cluster_rms.GetNbinsX()):
                pbin = cluster_rms.GetNbinsX()
            elif (pbin < 1):
                pbin = 1

            # Find CER
            cer = abs(cluster_rms.GetBinContent(pbin, ebin))
            if (p > 350):
                cer = 0.1

            # Apply smearing
            rng = np.random.default_rng()
            ptcer = cons_pt * (1 + rng.normal() * cer)

            # Calculate new energy
            Ecer = ptcer * np.cosh(cons_eta)
            # print("\nOld pT: {0:.4f}\tOld en: {1:.4f}".format(cons_pt, cons_en))
            # print("New pT: {0:0.4f}\tNew en: {1:.4f}".format(ptcer, Ecer))

            # Add new values to array builders
            p_builder.append(ptcer)
            E_builder.append(Ecer)

        ## Increment consituent counter
        constit_counter += 1

        ## If this is the last constituent in the jet, add array break
        if constit_counter == jet_constits:

            # Add array break
            p_builder.end_list()
            E_builder.end_list()
            p_builder.begin_list()
            E_builder.begin_list()

            # Reset constituent counter
            constit_counter = 0

            # Increment jet counter
            jet_counter += 1

    # End constituent loop
    # Take snapshots of builders
    var_pt = p_builder.snapshot()
    var_en = E_builder.snapshot()

    # Return dictionary with varied pT and energy information
    return {'fjet_clus_pt': var_pt, 'fjet_clus_E': var_en}
