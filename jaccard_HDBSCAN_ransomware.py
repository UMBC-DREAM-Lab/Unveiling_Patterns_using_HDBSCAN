#Script to compute jaccard distance and use HDBSCAN

import sys
import time
import heapq
import pickle
import ast
import csv
import random
import hdbscan
import datetime
import numpy as np
import pandas as pd
import multiprocessing
import functools
from UltraDict import UltraDict
from sklearn.utils.random import sample_without_replacement
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import jaccard_score
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse.csgraph import connected_components
from hdbscan.plots import SingleLinkageTree
from hdbscan._hdbscan_tree import condense_tree, compute_stability
from hdbscan._hdbscan_tree import get_clusters, outlier_scores
from hdbscan._hdbscan_linkage import label as slt_label
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs


def series_to_set(df_series):
    """Convert a series in a dataframe to a set, dropping -1"""
    s = ast.literal_eval(df_series)
    s.discard(-1)
    #if len(s) <= 10:
    if len(s) <= 10:
        return None
    return s

def convert_to_set(row):
    """Convert to a set"""
    try:
        #Convert string to list and remove empty entries
        lst = [int(item) for item in row.strip('[]').split(',') if item]
        return set(lst)
    except:
        return set()  # Return an empty set in case of error


#Applying Jaccard similarity
def get_file_similarities(idx_range):

    eps = 1e-5
    shm_file_rare_functions = UltraDict(name="file_rare_functions", create=False, shared_lock=True)
    shm_function_files = UltraDict(name="function_files", create=False, shared_lock=True)
    shm_function_matrix = UltraDict(name="function_matrix", create=False, shared_lock=True)

    start_idx, end_idx = idx_range
    dists = {}
    for file1_idx in range(start_idx, end_idx+1):
        
        if file1_idx % 1000 == 0:
            print(file1_idx, time.time())
            sys.stdout.flush()

        # Skip files with no rare functions
        if shm_file_rare_functions.get(file1_idx) is None:
            continue

        # Get set of rare function IDs in file
        rare_func_idxs = shm_file_rare_functions[file1_idx]

        # Build a set of files which share a rare function with file1
        file2_idxs = set()
        for func_idx in rare_func_idxs:
            file2_idxs.update(function_files[func_idx])

        # To avoid duplicate work, only consider files with an index greater than file1's
        file2_idxs = [idx for idx in file2_idxs if idx > file1_idx]

        # Compute jaccard distance between file1 and its related files
        file1_file2s = []
        file1_dists = []
        for file2_idx in file2_idxs:
            intersection = len(shm_function_matrix[file1_idx].intersection(shm_function_matrix[file2_idx]))
            union = len(shm_function_matrix[file1_idx].union(shm_function_matrix[file2_idx]))
            jaccard_similarity = intersection / union

            # Convert to distance and add very small value to keep distance nonzero
            # Only keep files with >= 95% similar set of functions
            if jaccard_similarity > 1:
                print(intersection, union, jaccard_similarity)
            if jaccard_similarity >= 0.95:
                jaccard_dist = 1 - jaccard_similarity + eps
                file1_file2s.append(file2_idx)
                file1_dists.append(jaccard_dist)
                if dists.get(file2_idx) is None:
                    dists[file2_idx] = []
                dists[file2_idx].append([file1_idx, jaccard_dist])
        dists[file1_idx] = list(zip(file1_file2s, file1_dists))

    return dists

#Calculate the distance for the pe_header information
def get_pe_feature_similarities(idx_range):
    eps = 1e-5
    shm_pe_data = UltraDict(name="file_pe_features", create=False, shared_lock=True)
    start_idx, end_idx = idx_range
    dists = {}

    for idx1 in range(start_idx, end_idx + 1):
        #set1 = pe_feature_data[idx1]

        if idx1 % 1000 == 0:
            print(idx1, time.time())
            sys.stdout.flush()
        #skip files with no pe data
        if shm_pe_data.get(idx1) is None:
            continue

        pe_data = shm_pe_data[idx1]

        file1_file2s = []
        file1_dists = []
        idx2_count = 0  # Counter

        # To avoid duplicate work, only consider files with an index greater than idx1's
        for idx2 in range(idx1 + 1, len(shm_pe_data)):
            if idx2_count >= 25:  # Stop adding more idx2 values once we reach 25
                break
            set2 = shm_pe_data[idx2]

            intersection = len(pe_data.intersection(set2))
            union = len(pe_data.union(set2))
            jaccard_similarity = intersection / union   # Can add eps to avoid division by zero

            if jaccard_similarity > 1:
                print(intersection, union, jaccard_similarity)

            if jaccard_similarity == 1:  # Applying a threshold
                jaccard_dist = 1 - jaccard_similarity + eps
                '''
                if dists.get(idx1) is None:
                    dists[idx1] = []
                if dists.get(idx2) is None:
                    dists[idx2] = []
                dists[idx1].append((idx2, jaccard_dist))
                dists[idx2].append((idx1, jaccard_dist))
'''
                file1_file2s.append(idx2)
                file1_dists.append(jaccard_dist)
                if dists.get(idx2) is None:
                    dists[idx2] = []
                dists[idx2].append([idx1, jaccard_dist])
                idx2_count += 1  # Increment the counter
        dists[idx1] = list(zip(file1_file2s, file1_dists))
        idx2_count = 0

    return dists


#Calculate the average of both Jaccard similarities between imports and pe headers.
def combine_ultradicts_batch(idx_range):

    shm_X_temp = UltraDict(name="X_temp", create=False, shared_lock=True)
    shm_X_pe = UltraDict(name="X_pe", create=False, shared_lock=True)
    start_idx, end_idx = idx_range
    partial_combined_dict = {}
    for key in range(start_idx, end_idx + 1):
        if key % 1000 == 0:
            print(key, time.time())
            sys.stdout.flush()
        if key in shm_X_temp or key in shm_X_pe:
            combined_values = []
            dict1 = {idx: dist for idx, dist in shm_X_temp.get(key, [])}
            dict2 = {idx: dist for idx, dist in shm_X_pe.get(key, [])}
            all_indices = set(dict1.keys()).union(dict2.keys())
            for idx in all_indices:
                dist1 = dict1.get(idx, 1)  # Default distance if not present
                dist2 = dict2.get(idx, 1)  # Default distance if not present
                averaged_distance = (dist1 + dist2) / 2
                #print(averaged_distance)
                combined_values.append((idx, averaged_distance))
            partial_combined_dict[key] = combined_values
    return partial_combined_dict


#Calcualte the reach scores
def get_reach_scores(idx_range):

    reach_scores = []
    file1_idxs = []
    file2_idxs = []

    start_idx, end_idx = idx_range
    shm_X = UltraDict(name="X", create=False, shared_lock=True)
    shm_core_dists = UltraDict(name="core_dists", create=False, shared_lock=True)
    for file1_idx in range(start_idx, end_idx):
        if file1_idx % 1000 == 0:
            print(file1_idx)
            sys.stdout.flush()
        if shm_X.get(file1_idx) is None:
            continue
        file2_info = shm_X[file1_idx]
        for file2_idx, dist in file2_info:
            reach_score = max([shm_core_dists[file1_idx], shm_core_dists[file2_idx], dist])
            reach_scores.append(reach_score)
            file1_idxs.append(file1_idx)
            file2_idxs.append(file2_idx)
    return reach_scores, file1_idxs, file2_idxs

for name in ["file_rare_functions", "function_files", "function_matrix", "X", "core_dists", "X_temp", "X_pe", "file_pe_features"]:
    UltraDict.unlink_by_name(name, ignore_errors=True)
    UltraDict.unlink_by_name(f'{name}_memory', ignore_errors=True)

#main function   
if __name__ == "__main__":

    print("Start", time.time())
    sys.stdout.flush()

    # Read dictionary pkl file
    with open('func_maps.pkl', 'rb') as fp:
        function = pickle.load(fp)

    #file_id_map = pd.read_csv('/data/results/prajna/results_ransomware_pe_info_final.csv', sep='\t')
    #file_id_map = pd.read_csv('/data/results/prajna/results_with_required_pe.csv', sep='\t')
    file_id_map = pd.read_csv('/data/results/prajna/results_pe_hopefully_final.csv', sep='\t')

    # Make the 'mapped_functions' column contain sets of function IDs
    function_matrix = file_id_map['mapped_functions'].apply(series_to_set)

    print(len(function_matrix))
    sys.stdout.flush()
    keep_df = file_id_map.iloc[[i for i, s in enumerate(function_matrix) if s is not None]]
    
    print(len(keep_df))

    print("Converted pandas dataframe to sets of function IDs", time.time())
    #print(keep_df.shape)
    print(len(keep_df))
    sys.stdout.flush()

    
    # Discard elements with < 10 imports
    function_matrix = [s for s in function_matrix if s is not None]

    # Randomly keep num_keep points in dataset
    '''
    num_keep = 100000
    rand_idxs = sample_without_replacement(len(function_matrix), num_keep)
    function_matrix = {i: function_matrix[idx] for i, idx in enumerate(rand_idxs)}
    keep_df = keep_df.iloc[rand_idxs]
    '''
    num_samples = len(function_matrix)
    # Comment this out if you uncomment the random keep
    function_matrix = {i: function_matrix[i] for i in range(num_samples)}


    print(keep_df.columns)
    md5s = keep_df["md5"].tolist()
    print(keep_df.keys())
    funcs = keep_df["dll_imports"].tolist()
    tokens = keep_df["av_tokens"].tolist()

    #Consider the pe informaation of only the keep_df
    # Extract the column data as a list of sets and store in Ultradict
    data_list = keep_df["mapped_pe_features"].apply(convert_to_set)
    
    try:
        data_list_dict = {i: s for i, s in enumerate(data_list)}
        buffer_size = sys.getsizeof(data_list) + 1000
        shm_pe_data = UltraDict(data_list_dict, name="file_pe_features",
                                            buffer_size=buffer_size, create=True, shared_lock=True)
    except UltraDict.Exceptions.AlreadyExists as e:
        print(f"Memory already exists: {e}")


    # Map each function to set of files it occurs in
    function_files = {}
    #num_samples = len(function_matrix)
    for file_idx in range(num_samples):
        for func_idx in function_matrix[file_idx]:
            if function_files.get(func_idx) is None:
                function_files[func_idx] = set()
            function_files[func_idx].add(file_idx)

    # Get list of functions that occur in less than some % of files
    rare_func_threshold = num_samples * 0.05 # 5% threshold
    rare_func_files = {}
    for func_idx, file_idxs in function_files.items():
        if len(file_idxs) <= rare_func_threshold:
            rare_func_files[func_idx] = file_idxs

    # Map files back to only their rare functions
    file_rare_functions = {}
    for func_idx, file_idxs in rare_func_files.items():
        for file_idx in file_idxs:
            if file_rare_functions.get(file_idx) is None:
                file_rare_functions[file_idx] = set()
            file_rare_functions[file_idx].add(func_idx)

    # Compare files which contain the same rare function
            
    eps = 1e-5 # Small value to add to each distance so it's nonzero
    print("Num samples", num_samples)

    #Handle the exception memory already exists
    try:
        # UltraDict for files -> rare functions
        buffer_size = sys.getsizeof(file_rare_functions) + 1000
        shm_file_rare_functions = UltraDict(file_rare_functions, name="file_rare_functions",
                                            buffer_size=buffer_size, create=True, shared_lock=True)

        # Ultradict for functions -> file IDs
        buffer_size = sys.getsizeof(function_files) + 1000
        shm_function_files = UltraDict(function_files, name="function_files",
                                    buffer_size=buffer_size, create=True, shared_lock=True)


        # Ultradict for file IDs -> functions
        buffer_size = sys.getsizeof(function_matrix) + 1000
        shm_function_matrix = UltraDict(function_matrix, name="function_matrix",
                                    buffer_size=buffer_size, create=True, shared_lock=True)

    except UltraDict.Exceptions.AlreadyExists as e:
        print(f"Memory already exists: {e}")

    # Compute jaccard distances in parallel
    pool = multiprocessing.Pool(32) # Use 32 cores
    batch_size = 1000
    start_idxs = list(range(0, num_samples, batch_size))
    end_idxs = list(range(batch_size-1, num_samples+batch_size-1, batch_size))
    end_idxs[-1] = num_samples-1
    idx_ranges = list(zip(start_idxs, end_idxs))


    results = pool.map(get_file_similarities, idx_ranges)

    print(len(results))

    # Construct nested dict storing distances
    X = {file1_idx: [] for file1_idx in range(num_samples)}
    for dists in results:
        for file1_idx, dist_info in dists.items():
            X[file1_idx].extend(dist_info)
    print("Finished jaccard distance calculations", time.time())
    print(len(X))
    print(sys.getsizeof(X))
    sys.stdout.flush()

    # Ultradict for all distances
    try:
        buffer_size = sys.getsizeof(X) + 1000
        shm_X_temp = UltraDict(X, name="X_temp", buffer_size=buffer_size, create=True, shared_lock=True)

    except UltraDict.Exceptions.AlreadyExists as e:
        print(f"Memory already exists: {e}")

    
    #Also compute jaccard similarity of pe information
    results_pe = pool.map(get_pe_feature_similarities, idx_ranges)

    print(len(results_pe))

    # Combine results
    combined_results = {file1_idx: [] for file1_idx in range(num_samples)}
    for dists in results_pe:
        for file1_idx, dist_info in dists.items():
            combined_results[file1_idx].extend(dist_info)
    print("Finished jaccard distance calculations for pe information", time.time())
    with open('check_combined_files.csv', 'w') as csv_file:  
        writer = csv.writer(csv_file)
        for key, value in combined_results.items():
            writer.writerow([key, value])
    print(len(combined_results))
    print(sys.getsizeof(combined_results))
    sys.stdout.flush()

    

    #Ultradict for distances for pe
    try:
        buffer_size = sys.getsizeof(combined_results) + 1000
        shm_X_pe = UltraDict(combined_results, name="X_pe", buffer_size=buffer_size, create=True, shared_lock=True)
        print("Finished storing in dictionary")
        sys.stdout.flush()

    except UltraDict.Exceptions.AlreadyExists as e:
        print(f"Memory already exists: {e}")


    #Combine both Ultradicts
    #CHECK: Try reducing the number of cores for these just incase this is where the memory is exceeding and the program is terminating
    pool = multiprocessing.Pool(16) # Tried 24 disn't work now trying 16   
    results_merge = pool.map(combine_ultradicts_batch, idx_ranges)
   
   # Combine results
    combined_ultradicts = {file1_idx: [] for file1_idx in range(num_samples)}
    for m_results in results_merge:
        for file1_idx, merge_results in m_results.items():
            combined_ultradicts[file1_idx].extend(merge_results)
    print("Finished jaccard distance calculations for commbining dictionaries", time.time())
    print(sys.getsizeof(combined_ultradicts))
    sys.stdout.flush()
    # Ultradict for all distances
    try:
        buffer_size = sys.getsizeof(combined_ultradicts) + 1000
        shm_X = UltraDict(combined_ultradicts, name="X", buffer_size=buffer_size, create=True, shared_lock=True)

    except UltraDict.Exceptions.AlreadyExists as e:
        print(f"Memory already exists: {e}")

    print("Completed all distance calculations")
    sys.stdout.flush()

    pool = multiprocessing.Pool(32) # Use 32 cores 
    # HDBSCAN hyperparameters
    k = 5
    min_samples = 10
    min_cluster_size = 10

    # Get core distance of each point
    core_dists = {}
    for file_idx in range(num_samples):
        if X.get(file_idx) is None:
            continue
        dists = [d for d, _ in X[file_idx]]
        n_smallest = [eps]
        if len(dists):
            n_smallest = heapq.nsmallest(k, dists)
        if len(n_smallest) > k:
            core_dists[file_idx] = n_smallest[k]
        else:
            core_dists[file_idx] = n_smallest[-1]

    try:
        buffer_size = sys.getsizeof(core_dists) + 1000
        shm_core_dists = UltraDict(core_dists, name="core_dists",
                               buffer_size=buffer_size, create=True, shared_lock=True)

    except UltraDict.Exceptions.AlreadyExists as e:
        print(f"Memory already exists: {e}")
        
    print("Computed all core distances", time.time())
    sys.stdout.flush()

    # Compute mutual reachablity distances between scans
    reach_scores = []
    file1_idxs = []
    file2_idxs = []
    results = pool.map(get_reach_scores, idx_ranges)
    for r, f1, f2 in results:
        #print("Enters this for loop to store values")
        reach_scores.extend(r)
        file1_idxs.extend(f1)
        file2_idxs.extend(f2)

    # Construct CSR matrix for reachability scores
    reach_matrix = csr_matrix((reach_scores, (file1_idxs, file2_idxs)), shape=(num_samples, num_samples))
    print("Constructed reachability matrix", time.time())
    sys.stdout.flush()

    # Compute Minimum Spanning Forest (MSF) of reachability matrix
    msf = minimum_spanning_tree(reach_matrix)
    msf_idxs = msf.nonzero()

    # Get all connected components in MSF
    msf_vals = msf[msf_idxs].A1
    msf_sparse = csr_matrix((msf_vals, msf_idxs), shape=(num_samples, num_samples))
    n_components, conn_labels = connected_components(msf_sparse)

    all_idxs = np.arange(num_samples)
    all_labels = np.array([-1]*num_samples)
    num_labels = 0

    # Iterate over each connected component
    for i in range(n_components):

        # Skip components smaller than min_cluster_size
        keep_idxs = all_idxs[conn_labels == i]
        if len(keep_idxs) < min_cluster_size:
            continue
        print("Component size:", len(keep_idxs))

        # Get Minimum Spanning Tree (MST) for component
        keep_rows = np.array([i for i, idx in enumerate(msf_idxs[0]) if idx in keep_idxs])
        mst_idxs = np.array(msf_idxs)[:, keep_rows]

        # label() will break if the set of idxs is not contiguous
        sorted_mst_idxs = sorted(keep_idxs)
        trans_idxs = {idx: i for i, idx in enumerate(sorted_mst_idxs)}

        # Sort MST edges by highest->lowest distance
        mst_vals = msf[mst_idxs[0], mst_idxs[1]][0].A1
        row_idxs = [trans_idxs[idx] for idx in mst_idxs[0]]
        col_idxs = [trans_idxs[idx] for idx in mst_idxs[1]]
        mst = np.vstack((row_idxs,) + (col_idxs,) + (mst_vals,)).T
        mst = mst[np.argsort(mst.T[2]), :]
        print("Got MST", time.time())
        sys.stdout.flush()

        # Get single linkage tree from MST
        slt = slt_label(mst)
        print("Got single linkage tree", time.time())
        sys.stdout.flush()

        # Condensing Single Linkage Tree and compute stability
        condensed_tree = condense_tree(slt, min_cluster_size)
        stability_dict = compute_stability(condensed_tree)
        print("Condensed tree, computed stability", time.time())
        sys.stdout.flush()

        # Get cluster labels
        cluster_labels, _, _ = get_clusters(
            condensed_tree,
            stability_dict,
            "leaf", # Selection Method
            True, # Allow single cluster
            False, # Match Reference implementation
            0.0, # Cluster selection epsilon
            0 # Max cluster size
        )
        print("Got cluster labels", time.time())
        sys.stdout.flush()

        # Update global labels with this clustering
        for i, label in enumerate(cluster_labels):
            if label != -1:
                label += num_labels
            idx = keep_idxs[i]
            all_labels[idx] = label
        num_labels += len(set(cluster_labels) - set([-1]))

    # TODO: Save to results folder!!!
    df_clusters = pd.DataFrame({
        'Cluster_Labels': all_labels,
        'md5s': md5s,
        'mapped_functions': funcs,
        'av_tokens': tokens,
    })

    #replace filename with something else
    #df_clusters.to_csv('/data/results/prajna/HDBSCAN_sparse_results_v3.csv')
    #df_clusters.to_csv('/data/results/prajna/HDBSCAN_sparse_results_v5.csv')
    df_clusters.to_csv('/data/results/prajna/HDBSCAN_sparse_results_rerun_v2.csv')

    print("Done")
    print(time.time())
    sys.stdout.flush()
