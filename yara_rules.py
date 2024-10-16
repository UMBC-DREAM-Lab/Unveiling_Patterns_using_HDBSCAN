#Scipt to generate YARA rules. 
#In order to generate the YARA rule we will have to define a rule.
#1) get all clusters and number of _id's in a cluster
#2) Select a cluster and check the functions that are similar
#3) Define the YARA rule

import sys
import time
import heapq
import pickle
import ast
import os
import datetime
import numpy as np
import pandas as pd
import multiprocessing


def generate_rule(hashes, imports, name, out_file):
    """
    Writes a YARA rule to out_file which matches files with the provided list
    of imported functions.

    Arguments:
    hashes -- List of file hashes
    imports -- List of (DLL, import) pairs
    name -- Name for the YARA rule
    out_file -- File to write the YARA rule to
    """

    rule = "import \"pe\"\n\nrule {} {{\n  meta:\n".format(name)
    for i in range(len(hashes)):
        rule += "    hash{} = \"{}\"".format(i+1, hashes[i])
    rule += "\n  condition:\n"
    for i, (dll, imp) in enumerate(imports):
        if i < len(imports) - 1:
            rule += "    pe.imports(\"{}\", \"{}\") and\n".format(dll, imp)
        else:
            rule += "    pe.imports(\"{}\", \"{}\")\n".format(dll, imp)
    rule += "}"
    #print(rule)
    with open(out_file, "w") as f:
        f.write(rule)

if __name__ == "__main__":

    output_folder = "/data/results/yara_rules_v3"

    #Read the file
    df_pe_info = pd.read_csv('/data/results/prajna/results_to_work_with.csv', sep='\t')

    #Code to pick only the required clusters
    specific_av_names = ['gandcrab', 'cerber', 'wannacry', 'cryptxxx', 'exxroute', 'tovicrypt','gotango', 'satan', 'titirez']
    av_name_df = df_pe_info[df_pe_info['av_name'].isin(specific_av_names)]

    unique_clusters = av_name_df['Cluster_Labels'].unique()

    for cluster_label in unique_clusters:
        # Focus on the current cluster
        cluster_data = av_name_df[av_name_df['Cluster_Labels'] == cluster_label]

        # Extract the 'mapped_functions' column for the current cluster
        mapped_functions_arrays = cluster_data['mapped_functions_y']

        # Convert the 'mapped_functions' arrays to sets for easy comparison
        mapped_functions_sets = [set(func) for func in mapped_functions_arrays]

        # Find the common elements in all arrays (intersection)
        common_functions = set(mapped_functions_sets[0]).intersection(*mapped_functions_sets[1:])

        #Adding a condition that the common functions should be greater than or equal to 6
        if len(common_functions) > 5:
            # Find the unique elements in each array
            unique_functions_per_array = [func_set - set(common_functions) for func_set in mapped_functions_sets]
        else:
            break

        # Extract data for YARA rule generation
        hashes = cluster_data['md5'].values
        dll_imports_str = cluster_data['dll_imports'].iloc[0]
        dll_imports_dict = ast.literal_eval(dll_imports_str)
        imports = [(dll, imp) for dll, imp_list in dll_imports_dict.items() for imp in imp_list]
       


        # Define the YARA rule name and output file name based on the cluster label
        rule_name = "cluster_{}".format(cluster_label)
        out_file = os.path.join(output_folder, "cluster_{}.yar".format(cluster_label))


        # Generate and write the YARA rule
        generate_rule(hashes, imports, rule_name, out_file)

        print(f"YARA rule generated for Cluster {cluster_label} and saved to {out_file}")

