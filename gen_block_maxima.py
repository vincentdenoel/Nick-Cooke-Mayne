import pandas as pd
import dask.dataframe as dd
import os
import numpy as np
from gen_sample import gen_sample
import nicks_functions as nf

import time

def gen_block_maxima(n_blocks=10, series_types=[0, 1, 2, 3], k_blocks=[0, 2, 4], unique_stamp=None):
    """
    Generate block maxima for different series and save them to parquet files.
    
    Parameters:
    n_blocks (int): Number of blocks to divide the time series into.
    """
    
    # Initialize or load data
    dothis = True  # Perform the Monte Carlo simulation (could be long)
    saveStep = 50
    parquet_dir = 'maximas_data'
    T_base = 6000  # Base time series length
    block_base = 10  # Base block sizes
    spinup_blocks = 1
    nbRuns = 10000 * block_base // n_blocks  # Number of Monte Carlo runs
    T = T_base * (n_blocks + spinup_blocks) / block_base
    if unique_stamp is None:
        unique_stamp = int(time.time())

    print(f"n_blocks: {n_blocks}, nbRuns: {nbRuns}")

    # Create directory if it doesn't exist
    os.makedirs(parquet_dir, exist_ok=True)

    if dothis:
        start_time = time.time()
        k_start = 0

        maximas = {}
        for i in series_types:
            maximas[f"series{i}"] = {}
            for k_block in k_blocks:
                if k_block == 0:
                    maximas[f"series{i}"][f"SBM"] = []
                else:
                    maximas[f"series{i}"][f"CBM_k{k_block}"] = []
    
        for k in range(k_start, nbRuns):
            for i in series_types:
                x, t, dt = gen_sample(i, T=T)
                start_index = int(len(x) * spinup_blocks / (n_blocks + spinup_blocks))
                x = x[start_index:]
                t = t[start_index:]
                block_max = lambda x: np.max(x, axis=-1)
            
                for k_block in k_blocks:
                    if k_block == 0:
                        maximas[f"series{i}"][f"SBM"].append(nf.sample_blocks(x, len(x)//n_blocks, random=False, circular=True, step = 1, func=block_max))
                    else:
                        r = int(len(x)/n_blocks/k_block)
                        maximas[f"series{i}"][f"CBM_k{k_block}"].append(nf.CBM(x, r, k = k_block, circular=False, step = 1).flatten())
        
            # Save data every saveStep iterations
            if (k + 1) % saveStep == 0:
                for i in series_types:
                    combined = {}
                    for key, data_list in maximas[f"series{i}"].items():
                        if data_list:  # Only process non-empty lists
                            stacked = np.vstack(data_list)
                            values = stacked.ravel()
                            combined[key] = values
                            maximas[f"series{i}"][key] = []

                    df_long = pd.DataFrame(combined)

                    # Manually adjust the index based on how many rows we've already written
                    series_name = f"series{i}"

                    ddf = dd.from_pandas(df_long, npartitions=saveStep//25)

                    parquet_file = os.path.join(parquet_dir, f"{series_name}-{unique_stamp}.parquet")

                    if os.path.exists(parquet_file):
                        ddf.to_parquet(parquet_file, compression="zstd", append=True, write_index=False, write_metadata_file=True)
                    else:
                        ddf.to_parquet(parquet_file, compression="zstd", write_index=False, write_metadata_file=True,)
            
                    print(f"Saved data at iteration {k}")

            elapsed_time = time.time() - start_time
            print(f"Entry {k}, time {elapsed_time:.2f} seconds")
            start_time = time.time()

def main():
    """
    Main function to execute the block maxima generation.
    """
    import argparse
    parser = argparse.ArgumentParser(description='Generate block maxima for time series')
    parser.add_argument('--series_type', type=int, help='Type of series to generate (0-3)')
    parser.add_argument('--unique_stamp', type=int, help='Unique stamp for file naming')
    args = parser.parse_args()

    n_blocks = 10  # Number of blocks
    series_types = [args.series_type] if args.series_type is not None else [0, 1]
    unique_stamp = args.unique_stamp if args.unique_stamp is not None else None
    k_blocks = [0, 2, 4]
    print(f"series_types: {series_types}, k_blocks: {k_blocks}")
    gen_block_maxima(n_blocks, series_types=series_types, k_blocks=k_blocks, unique_stamp=unique_stamp)

if __name__ == "__main__":
    main()