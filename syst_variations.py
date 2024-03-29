""" syst_variations.py - This script defines function which apply systematic
variations to the constituent level inputs. These functions can be passed to
the root converter class to produce data set with applied systematics.

Currently, the variations implemented are the EM scale cluster uncertainties.

Author: Kevin Greif
Last updated 7/19/22
python3
"""

import awkward as ak
import numpy as np

def reco_efficiency(jets, uncert_map):
    """ reco_efficiency - This function applies the cluster reconstruction
    efficiency systematic variation to the constituent level inputs contained
    in the jet batch.

    Arguments:
    jets (dict): The jet batch, almost always as defined in the root converter
    class after cuts have been applied. See root_converter.py for details.
    uncert_map (TFile): The uncertaintiy map file object loaded using PyROOT

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

        ## If constituent is not neutral (taste == 0), write True
        if cons_taste == 0:
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
    constit_branches = [
        'fjet_clus_pt', 'fjet_clus_eta', 'fjet_clus_phi', 'fjet_clus_E',
        'fjet_clus_taste'
    ]
    var_dict = {kw: jets[kw][keep] for kw in constit_branches}

    print("For this batch we dropped {0:f} percent of constituents".format(dropped_counter * 100 / total_counter))

    return var_dict


def energy_scale(jets, uncert_map, direction='up'):
    """ energy_scale - This function applies the cluster energy scale
    variation to the constituent level inputs contained in the jet batch.

    Function assumes pT and energy are given in MeV

    Arguments:
    jets (dict): The jet batch, almost always as defined in the root converter
    class after cuts have been applied. See root_converter.py for details.
    uncert_map (TFile): The uncertaintiy map file object loaded using PyROOT
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
    pt = jets['fjet_clus_pt'] / 1000
    en = jets['fjet_clus_E'] / 1000

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

        ## If constituent is charged, write nominal values
        if cons_taste == 0:

            # Write nominal, remembering to convert back to MeV
            p_builder.append(1000 * cons_pt)
            E_builder.append(1000 * cons_en)

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
            pbin = cluster_means.GetXaxis().FindBin(p)

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

            # Apply pT and energy variation
            if direction == 'up':
                ptces = cons_pt * (1 + ces)
                Eces = cons_en * (1 + ces)
            elif direction == 'down':
                ptces = cons_pt * (1 - ces)
                Eces = cons_en * (1 - ces)
            else:
                raise ValueError("Direction must be up or down!")

            # print("\nOld pT: {0:.4f}\tOld en: {1:.4f}".format(cons_pt, cons_en))
            # print("New pT: {0:0.4f}\tNew en: {1:.4f}".format(ptces, Eces))

            # Add new values to array builders, remembering to convert back to MeV
            p_builder.append(1000 * ptces)
            E_builder.append(1000 * Eces)

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

def energy_res(jets, uncert_map):
    """ energy_res - This function applies the cluster energy resolution
    variation to the constituent level inputs. Assumes constituent pT and
    E values are given in units of MeV

    Arguments:
    jets (dict): The jet batch, almost always as defined in the root converter
    class after cuts have been applied. See root_converter.py for details.
    uncert_map (TFile): The uncertaintiy map file object loaded using PyROOT

    Returns:
    (dict): A dictionary containing the constituent level quantities with
    applies systematic variation.
    """

    # Get cluster energy scale and RMS from uncert map
    cluster_scale = uncert_map.Get('Scale')
    cluster_rms = uncert_map.Get('RMS')

    # Pull pt and energy
    pt = jets['fjet_clus_pt'] / 1000
    en = jets['fjet_clus_E'] / 1000

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
            
    # Initialize random number generator
    rng = np.random.default_rng()

    # Constituent loop
    for cons_en, cons_eta, cons_pt, cons_taste in iterable:

        ## Start by finding number of constits in a jet, only if this is
        # the first constituent
        if constit_counter == 0:
            jet_constits = n_constits[jet_counter]

        ## If constituent is charged, write nominal values
        if cons_taste == 0:

            # Write nominal, converting back to MeV
            p_builder.append(1000 * cons_pt)
            E_builder.append(1000 * cons_en)

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

            # Calculate smearing factor
            factor = 1 + rng.normal(loc=0.0, scale=1.0) * cer
            
            # Apply smearing
            ptcer = factor * cons_pt
            Ecer = factor * cons_en

            # print("\nOld pT: {0:.4f}\tNew pT: {1:.4f}".format(cons_pt, ptcer))
            # print("Old en: {0:0.4f}\tNew en: {1:.4f}".format(cons_en, Ecer))
            # print("Percent variation: {0:.4f}".format((Ecer / cons_en) - 1))

            # Add new values to array builders
            p_builder.append(1000 * ptcer)
            E_builder.append(1000 * Ecer)

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


def position(jets, uncert_map):
    """ position - This function applies the cluster position resolution
    variation to the constituent level inputs. Assumes eta and phi
    information is given in units of radians.

    Arguments:
    jets (dict): The jet batch, almost always as defined in the root converter
    class after cuts have been applied. See root_converter.py for details.
    uncert_map (TFile): The uncertaintiy map file object loaded using PyROOT

    Returns:
    (dict): A dictionary containing the constituent level quantities with
    applies systematic variation.
    """

    # Pull information from batch
    eta = ak.flatten(jets['fjet_clus_eta'])
    phi = ak.flatten(jets['fjet_clus_phi'])
    pt = ak.flatten(jets['fjet_clus_pt'])

    # Instead of building an awkard array with boolean values, we directly
    # build the new pT and energy values for the constituents
    n_jets = len(jets['fjet_clus_pt'])
    n_constits = ak.count(jets['fjet_clus_pt'], axis=1)
    total_constits = ak.sum(n_constits)

    ## Initialize awkward array builders to construct new 4 vector
    eta_builder = ak.ArrayBuilder()
    phi_builder = ak.ArrayBuilder()
    pt_builder = ak.ArrayBuilder()

    # Immediately begin list in array builders
    eta_builder.begin_list()
    phi_builder.begin_list()
    pt_builder.begin_list()

    ## Loop over flattened array, putting list breaks in awkward array based
    # on information in n_constits vector
    taste = ak.flatten(jets['fjet_clus_taste'])
    iterable = zip(eta, phi, pt, taste) 

    # Counters to manage list breaks
    jet_counter = 0
    constit_counter = 0

    # Initialize random number generator
    rng = np.random.default_rng()

    # Constituent loop
    for cons_eta, cons_phi, cons_pt, cons_taste in iterable:

        ## Start by finding number of constits in a jet, only if this is
        # the first constituent
        if constit_counter == 0:
            jet_constits = n_constits[jet_counter]

        ## If constituent is charged or merged, write nominal values
        if cons_taste == 0 or cons_taste == 2:

            # Write nominal, converting back to MeV
            eta_builder.append(cons_eta)
            phi_builder.append(cons_phi)
            pt_builder.append(cons_pt)

        # Else, we apply systematic variation
        else:

            # Smear eta and phi
            eta_smeared = rng.normal(loc=cons_eta, scale=0.005)
            phi_smeared = rng.normal(loc=cons_phi, scale=0.005)

            # Calculate new pT
            pt_smeared = cons_pt * np.cosh(cons_eta) / np.cosh(eta_smeared)

            # print("\nOld eta: {0:.4f}\tNew eta: {1:.4f}".format(cons_eta, eta_smeared))
            # print("Old phi: {0:0.4f}\tNew phi: {1:.4f}".format(cons_phi, phi_smeared))
            # print("Old pT: {0:.4f}\tNew pT: {1:.4f}".format(cons_pt, pt_smeared))

            # Add new values to array builders
            eta_builder.append(eta_smeared)
            phi_builder.append(phi_smeared)
            pt_builder.append(pt_smeared)

        ## Increment consituent counter
        constit_counter += 1

        ## If this is the last constituent in the jet, add array break
        if constit_counter == jet_constits:

            # Add array break
            eta_builder.end_list()
            phi_builder.end_list()
            pt_builder.end_list()
            eta_builder.begin_list()
            phi_builder.begin_list()
            pt_builder.begin_list()

            # Reset constituent counter
            constit_counter = 0

            # Increment jet counter
            jet_counter += 1

    # End constituent loop
    # Take snapshots of builders
    var_eta = eta_builder.snapshot()
    var_phi = phi_builder.snapshot()
    var_pt = pt_builder.snapshot()

    # Return dictionary with varied 4-vector
    return {'fjet_clus_eta': var_eta,
            'fjet_clus_phi': var_phi,
            'fjet_clus_pt': var_pt}

