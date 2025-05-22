import pandas as pd
import numpy as np
from MDAnalysis import Universe
from MDAnalysis.analysis.distances import distance_array
from calc_df import *
from utils import *
from scipy.spatial import cKDTree
from collections import Counter
import networkx as nx
import os


def get_aggregates(grp1, grp2, box,pep_len):
    # Determine which of the above peptides are aggregates
    pep_aggregates = np.where(distance_array(grp1,grp2,box)<7)
    agg_x = pep_aggregates[0]//pep_len
    agg_y = pep_aggregates[1]//pep_len
    
    # Remove self-interactions
    pairs = [(int(i), int(j)) for i, j in zip(agg_x, agg_y) if i != j]
    
    # Order pairs so that (i,j) is equivalent to (j,i)
    normalized_pairs = [tuple(sorted(pair)) for pair in pairs]  
    
    # Now get counts and create Graph 
    pair_counts = Counter(normalized_pairs)
    interacting_pairs = [ pair for pair, count in pair_counts.items() if count>4]    
    
    G = nx.Graph()
    G.add_nodes_from(range(36))  # assuming 36 peptides always
    G.add_edges_from(interacting_pairs)
    S = [c for c in sorted(nx.connected_components(G), key=len)]

    return S


# Function to get APL values from CSV files
def get_apl_values(frame, file_pair):
    if file_pair == 0:
        filename = f'/Users/nehananajkar/Desktop/Lab/resub_traj/APL_DAT/II_frame{frame}_frame_00000.csv'
        return filename
    elif file_pair == 1:
        filename = f'/Users/nehananajkar/Desktop/Lab/resub_traj/APL_DAT/III_frame{frame}_frame_00000.csv'
        return filename
    elif file_pair == 2:
        filename = f'/Users/nehananajkar/Desktop/Lab/resub_traj/APL_DAT/IV_frame{frame}_frame_00000.csv'
        return filename
    else:
        print('Couldnt find APL file for frame{frame} and traj {file_pair}')
        return None

def get_closest_lipids_resid(peptide_positions, lipid_positions, lipid_resids, k=10):
    """
    Find the k closest lipid atoms (or beads) to a peptide and return their resid values.

    Parameters:
        peptide_positions (np.ndarray): N x 3 array of peptide atom/bead positions.
        lipid_positions (np.ndarray): M x 3 array of lipid atom/bead positions.
        lipid_resids (np.ndarray): M array of lipid residue indices.
        k (int): Number of closest lipids to return.

    Returns:
        closest_positions (np.ndarray): k x 3 array of closest lipid positions.
        closest_resids (np.ndarray): Resid indices of the closest lipids.
    """
    lipid_tree = cKDTree(lipid_positions)
    dists, indices = lipid_tree.query(peptide_positions, k=k)

    if len(peptide_positions) > 1:
        indices = np.unique(indices.flatten())[:k]
    else:
        indices = np.array(indices).flatten()[:k]

    closest_positions = lipid_positions[indices]
    closest_resids = lipid_resids[indices]
    return closest_positions, closest_resids