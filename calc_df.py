import pandas as pd
import numpy as np
from MDAnalysis import Universe
from MDAnalysis.analysis.distances import distance_array
import networkx as nx

def _isincircle(point, center, radius):
	## Given the x and y coords, checks if a bead is within the curved region
	pos = np.sqrt((point[0]-center[0])**2 + (point[1]-center[1])**2)
	if pos>radius:
		return False
	else:
		return True

def _sort_lipids(po4, box):
    '''
    Sorts po4s into curved and planar
    Returns AtomGroup for curved and planar po4s

    '''
    x = box[0]
    y = box[1]

    center = [x/2.0,y/2.0]

    a = 0
    curved_po4 =[]
    planar_po4 = []
    for i in po4:
        if _isincircle(i.position,center,75):
            curved_po4.append(a)
        else:
            planar_po4.append(a)
        a=a+1

    return po4[np.array(curved_po4)], po4[np.array(planar_po4)]



def pep_pep_contacts(df, traj_files):
    """
    Calculate the number of peptide–peptide hydrogen-like contacts
    for selected peptides (from DataFrame) across multiple trajectories.

    Parameters:
    - df: DataFrame with columns ['trajectory', 'frame', 'peptideIndex']
    - traj_files: dict mapping trajectory index -> (tpr, xtc)

    Returns:
    - contact_df: DataFrame with frame, trajectory, peptideIndex, pp_contacts
    """
    contact_records = []

    for traj_idx, group in df.groupby("trajectory"):
        tpr, xtc = traj_files[traj_idx]
        u = Universe(tpr, xtc)
        prot = u.select_atoms("name BB S1 S2 S3 S4")

        num_peptides = 36
        pep_len = len(prot) // num_peptides
        assert len(prot) % num_peptides == 0, "Peptide division mismatch."

        all_peptides = [prot[i * pep_len : (i + 1) * pep_len] for i in range(num_peptides)]
        
        frames_to_check = group["frame"].unique()
        peptide_lookup = group.groupby("frame")["local_pep_ID"].apply(list).to_dict()

        for ts in u.trajectory:
            if ts.frame not in frames_to_check:
                continue
            active_peptides = peptide_lookup[ts.frame]

            for i in active_peptides:
                peptide_i = all_peptides[i]  
                count = 0

                for j in range(num_peptides):
                    if j == i:
                        continue
                    peptide_j = all_peptides[j]
                    # There is double counting ( i-j contacts and j-i contacts)
                    d_pp = distance_array(peptide_i.positions, peptide_j.positions, box=u.dimensions)
                    count = (d_pp < 7).sum()

                contact_records.append({
                    'trajectory': traj_idx,
                    'frame': ts.frame,
                    'local_pep_ID': i,
                    'pp_contacts': count
                })

    return pd.DataFrame(contact_records)


def build_init_df(tpr, xtc, traj_index, start=0, stop=None) -> pd.DataFrame:
    u = Universe(tpr, xtc)
    po4 = u.select_atoms('name PO4')
    prot = u.select_atoms('name BB S1 S2 S3 S4')

    num_peptides = 36
    pep_len = int(len(prot) / num_peptides)
    assert len(prot) % num_peptides == 0, "Peptide division mismatch."

    peptides = [prot[i * pep_len: (i + 1) * pep_len] for i in range(num_peptides)]
    records = []

    for ts in u.trajectory[start:stop]:
        curved_po4, planar_po4 = _sort_lipids(po4, u.dimensions)

        peptide_status = {}

        # Step 1: Identify direct membrane contacts
        direct_membrane_peptides = set()
        for i, peptide in enumerate(peptides):
            d_curved = distance_array(peptide.positions, curved_po4.positions, box=u.dimensions)
            d_planar = distance_array(peptide.positions, planar_po4.positions, box=u.dimensions)
            n_curved = (d_curved < 7.0).sum()
            n_planar = (d_planar < 7.0).sum()

            if n_curved >= 5:
                peptide_status[i] = 0
                direct_membrane_peptides.add(i)
            elif n_planar >= 5:
                peptide_status[i] = 1
                direct_membrane_peptides.add(i)
            else:
                peptide_status[i] = -1  # undecided for now, could be class (2/3/solution peptides)

        # Step 2: Build peptide-peptide interaction graph
        contact_graph = nx.Graph()
        contact_graph.add_nodes_from(range(num_peptides))
        for i in range(num_peptides):
            for j in range(i + 1, num_peptides):
                d_pp = distance_array(peptides[i].positions, peptides[j].positions, box=u.dimensions)
                if (d_pp < 7.0).sum() >= 5:
                    contact_graph.add_edge(i, j)

        # Step 3: Propagate indirect labels
        for component in nx.connected_components(contact_graph):
            # Check if any peptide in this component is directly bound
            direct_in_component = direct_membrane_peptides.intersection(component)
            if direct_in_component:
                # Assign class 2 or 3 based on which direct peptide is closest
                for pid in component:
                    if peptide_status[pid] in [0, 1]:
                        continue  # already assigned
                    # Pick closest direct peptide
                    for ref_id in direct_in_component:
                        ref_class = peptide_status[ref_id]
                        if ref_class == 0:
                            peptide_status[pid] = 2
                        elif ref_class == 1:
                            peptide_status[pid] = 3
                        break  # Assign first match and break

        # Step 4: Record peptides with assigned class
        for i in range(num_peptides):
            if peptide_status[i] == -1: # -1 being peptides in solution
                continue  # skip unclassified (solution)
            global_id = traj_index * num_peptides + i
            records.append({
                'global_pep_ID': global_id,
                'local_pep_ID': i,
                'frame': ts.frame,
                'trajectory': traj_index,
                'target_status': peptide_status[i]
            })

    return pd.DataFrame(records)

def pep_lip_contacts(df, traj_files):

    contact_records = []

    for traj_idx, group in df.groupby("trajectory"):
        tpr, xtc = traj_files[traj_idx]
        u = Universe(tpr, xtc)
        prot = u.select_atoms("name BB S1 S2 S3 S4")
        lip = u.select_atoms("resname POPC and name PO4")
        num_peptides = 36
        pep_len = len(prot) // num_peptides
        assert len(prot) % num_peptides == 0, "Peptide division mismatch."

        all_peptides = [prot[i * pep_len : (i + 1) * pep_len] for i in range(num_peptides)]
        
        frames_to_check = group["frame"].unique()
        peptide_lookup = group.groupby("frame")["local_pep_ID"].apply(list).to_dict()
        # Considering all peptides that may be interating with the membrane, not just direct interactions
        for ts in u.trajectory:
            if ts.frame not in frames_to_check:
                continue
            active_peptides = peptide_lookup[ts.frame]
            for i in active_peptides:
                peptide_i = all_peptides[i]
                
                d_pp = distance_array(peptide_i.positions, lip.positions, box=u.dimensions)
                count = (d_pp < 7.0).sum()

                contact_records.append({
                    'trajectory': traj_idx,
                    'frame': ts.frame,
                    'local_pep_ID': i,
                    'pl_contacts': count
                })

    return pd.DataFrame(contact_records)

def dip_pep_pep_contacts(df, traj_files):

    contact_records = []

    for traj_idx, group in df.groupby("trajectory"):
        tpr, xtc = traj_files[traj_idx]
        u = Universe(tpr, xtc)
        
        prot = u.select_atoms("name BBp BBm")
        num_peptides = 36
        pep_len = len(prot) // num_peptides
        assert len(prot) % num_peptides == 0, "Peptide division mismatch."

        all_peptides = [prot[i * pep_len : (i + 1) * pep_len] for i in range(num_peptides)]
        
        frames_to_check = group["frame"].unique()
        peptide_lookup = group.groupby("frame")["local_pep_ID"].apply(list).to_dict()

        for ts in u.trajectory:
            if ts.frame not in frames_to_check:
                continue
            active_peptides = peptide_lookup[ts.frame]

            for i in active_peptides:
                peptide_i = all_peptides[i]  
                count = 0

                for j in range(num_peptides):
                    if j == i:
                        continue
                    peptide_j = all_peptides[j]
                    # There is double counting ( i-j contacts and j-i contacts)
                    d_pp = distance_array(peptide_i.positions, peptide_j.positions, box=u.dimensions)
                    count = (d_pp < 3.5).sum()

                contact_records.append({
                    'trajectory': traj_idx,
                    'frame': ts.frame,
                    'local_pep_ID': i,
                    'dip_pp_contacts': count
                })

    return pd.DataFrame(contact_records)

def dip_pep_lip_contacts(df, traj_files):

    contact_records = []

    for traj_idx, group in df.groupby("trajectory"):
        tpr, xtc = traj_files[traj_idx]
        u = Universe(tpr, xtc)
        prot = u.select_atoms("name BBp BBm")
        lip = u.select_atoms("resname POPC")
        num_peptides = 36
        pep_len = len(prot) // num_peptides
        assert len(prot) % num_peptides == 0, "Peptide division mismatch."

        all_peptides = [prot[i * pep_len : (i + 1) * pep_len] for i in range(num_peptides)]
        
        frames_to_check = group["frame"].unique()
        peptide_lookup = group.groupby("frame")["local_pep_ID"].apply(list).to_dict()
        # Considering all peptides that may be interating with the membrane, not just direct interactions
        for ts in u.trajectory:
            if ts.frame not in frames_to_check:
                continue
            active_peptides = peptide_lookup[ts.frame]
            for i in active_peptides:
                peptide_i = all_peptides[i]
                
                d_pp = distance_array(peptide_i.positions, lip.positions, box=u.dimensions)
                count = (d_pp < 3.5).sum()

                contact_records.append({
                    'trajectory': traj_idx,
                    'frame': ts.frame,
                    'local_pep_ID': i,
                    'dip_lp_contacts': count
                })

    return pd.DataFrame(contact_records)


def hyd_pep_lip_contacts(df, traj_files)->pd.DataFrame:

    contact_records = []

    for traj_idx, group in df.groupby("trajectory"):
        tpr, xtc = traj_files[traj_idx]
        u = Universe(tpr, xtc)
        prot = u.select_atoms("resname PHE ILE LEU VAL MET and name S1 S2 S3")
        lip = u.select_atoms("resname POPC and name C* D*")
        num_peptides = 36
        pep_len = len(prot) // num_peptides
        assert len(prot) % num_peptides == 0, "Peptide division mismatch."

        all_peptides = [prot[i * pep_len : (i + 1) * pep_len] for i in range(num_peptides)]
        
        frames_to_check = group["frame"].unique()
        peptide_lookup = group.groupby("frame")["local_pep_ID"].apply(list).to_dict()
        # Considering all peptides that may be interating with the membrane, not just direct interactions
        for ts in u.trajectory:
            if ts.frame not in frames_to_check:
                continue
            active_peptides = peptide_lookup[ts.frame]
            for i in active_peptides:
                peptide_i = all_peptides[i]
                
                d_pp = distance_array(peptide_i.positions, lip.positions, box=u.dimensions)
                count = (d_pp < 7.0).sum()

                contact_records.append({
                    'trajectory': traj_idx,
                    'frame': ts.frame,
                    'local_pep_ID': i,
                    'hyd_lp_contacts': count
                })

    return pd.DataFrame(contact_records)

def Ppep_Nlip_contacts(df, traj_files)->pd.DataFrame:

    contact_records = []

    for traj_idx, group in df.groupby("trajectory"):
        tpr, xtc = traj_files[traj_idx]
        u = Universe(tpr, xtc)
        prot = u.select_atoms("resname LYS ARG and name S2")
        lip = u.select_atoms("resname POPC and name PO4")
        num_peptides = 36
        pep_len = len(prot) // num_peptides
        assert len(prot) % num_peptides == 0, "Peptide division mismatch."

        all_peptides = [prot[i * pep_len : (i + 1) * pep_len] for i in range(num_peptides)]
        
        frames_to_check = group["frame"].unique()
        peptide_lookup = group.groupby("frame")["local_pep_ID"].apply(list).to_dict()
        # Considering all peptides that may be interating with the membrane, not just direct interactions
        for ts in u.trajectory:
            if ts.frame not in frames_to_check:
                continue
            active_peptides = peptide_lookup[ts.frame]
            for i in active_peptides:
                peptide_i = all_peptides[i]
                
                d_pp = distance_array(peptide_i.positions, lip.positions, box=u.dimensions)
                count = (d_pp < 7.0).sum()

                contact_records.append({
                    'trajectory': traj_idx,
                    'frame': ts.frame,
                    'local_pep_ID': i,
                    'Ppep_Nlip_contacts': count
                })

    return pd.DataFrame(contact_records)

def Npep_Plip_contacts(df, traj_files)->pd.DataFrame:

    contact_records = []

    for traj_idx, group in df.groupby("trajectory"):
        tpr, xtc = traj_files[traj_idx]
        u = Universe(tpr, xtc)
        prot = u.select_atoms("resname ASP GLU and name S2")
        lip = u.select_atoms("resname POPC and name NC3")
        num_peptides = 36
        pep_len = len(prot) // num_peptides
        assert len(prot) % num_peptides == 0, "Peptide division mismatch."

        all_peptides = [prot[i * pep_len : (i + 1) * pep_len] for i in range(num_peptides)]
        
        frames_to_check = group["frame"].unique()
        peptide_lookup = group.groupby("frame")["local_pep_ID"].apply(list).to_dict()
        # Considering all peptides that may be interating with the membrane, not just direct interactions
        for ts in u.trajectory:
            if ts.frame not in frames_to_check:
                continue
            active_peptides = peptide_lookup[ts.frame]
            for i in active_peptides:
                peptide_i = all_peptides[i]
                
                d_pp = distance_array(peptide_i.positions, lip.positions, box=u.dimensions)
                count = (d_pp < 7.0).sum()

                contact_records.append({
                    'trajectory': traj_idx,
                    'frame': ts.frame,
                    'local_pep_ID': i,
                    'Npep_Plip_contacts': count
                })

    return pd.DataFrame(contact_records)
