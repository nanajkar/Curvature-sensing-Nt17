import pandas as pd
import numpy as np
from MDAnalysis import Universe
from MDAnalysis.analysis.distances import distance_array
from calc_df import *
from utils import *
from scipy.interpolate import griddata
from scipy.spatial import KDTree
from typing import Tuple

"""
Functions that calculate temporal values for the df
"""


def _create_membrane_surface(PO4_positions:np.array)->Tuple[scipy.spatial.KDTree,np.array]: 
    """
    Constructs an interpolated membrane surface from PO4 positions
    """
    # Assuming the membrane is roughly parallel to the XY plane
    x = PO4_positions[:, 0]
    y = PO4_positions[:, 1]
    z = PO4_positions[:, 2]
    
    # Create a grid to interpolate the membrane surface
    xi, yi = np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100)
    xi, yi = np.meshgrid(xi, yi)
    
    # Interpolate to get a smooth surface
    zi = griddata((x, y), z, (xi, yi), method='cubic')
    zi = np.nan_to_num(zi, nan=0.0, posinf=np.max(z), neginf=np.min(z))
    membrane_surface_points = np.c_[xi.ravel(), yi.ravel(), zi.ravel()]
    kdtree = KDTree(membrane_surface_points)
    
    return kdtree, membrane_surface_points

def _calculate_insertion_depth(molecule_positions, kdtree, membrane_surface_points)->np.float:
    """
    Calculate the minimum insertion depth(A) of a molecule relative to the membrane surface.
    """
    distances = []
    for pos in molecule_positions:
        _, idx = kdtree.query(pos)
        nearest_surface_point = membrane_surface_points[idx]
        distance = np.linalg.norm(pos - nearest_surface_point)
        distances.append(distance)
    return np.min(distances)
    
    
def calc_separation_values(df, traj_files)->pd.DataFrame:
        
    records = []
    for traj_idx, group in df.groupby("trajectory"):
        pep_of_interest = group['local_pep_ID'].unique()
        tpr, xtc = file_pairs[traj_idx]
        u = Universe(tpr, xtc)
        
        prot = u.select_atoms('name BB S1 S2 S3')
        num_peptides = 36
        pep_len = len(prot) // num_peptides
        assert len(prot) % num_peptides == 0, "Peptide division mismatch."
        
        po4 = u.select_atoms('resname POPC and name PO4')
        all_peptides = [prot[i * pep_len : (i + 1) * pep_len] for i in range(num_peptides)]

        # Prepare storage: one list per peptide
        peptide_distances = {pep_id: [] for pep_id in pep_of_interest}

        # Now loop through trajectory once
        for ts in u.trajectory[:5001]:
            po4_pos = po4.positions
            grid, membrane_surface_points = _create_membrane_surface(po4_pos)

            for pep_id in pep_of_interest:
                min_dist = _calculate_insertion_depth(all_peptides[pep_id].positions, grid, membrane_surface_points)
                peptide_distances[pep_id].append(min_dist)

        # Now postprocess
        for peptide in pep_of_interest:
            all_dist_vals = peptide_distances[peptide]
            frames_of_interest = group.loc[group['local_pep_ID'] == peptide, 'frame']

            for frame in frames_of_interest:
                mean_val = np.mean(all_dist_vals[:frame+1])
                records.append({
                    'trajectory': traj_idx,
                    'frame': frame,
                    'local_pep_ID': peptide,
                    'mean_separation': mean_val
                })

    # Optional: Create final DataFrame
    mean_separation_df = pd.DataFrame(records)
    
    return mean_separation_df
