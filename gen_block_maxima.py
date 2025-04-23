import pandas as pd
import dask.dataframe as dd
import os
import numpy as np
from gen_sample import gen_sample
import nicks_functions as nf

import time

def gen_block_maxima(n_blocks=10, series_types=[0, 1, 2, 3], k_blocks=[0, 2, 4]):
    """
    Generate block maxima for different series and save them to parquet files.
    
    Parameters:
    n_blocks (int): Number of blocks to divide the time series into.
    """
    
    # Initialize or load data
    dothis = True  # Perform the Monte Carlo simulation (could be long)
    nbRuns = 10000  # Number of Monte Carlo runs
    saveStep = 100
    parquet_dir = 'maximas_data'

    # Create directory if it doesn't exist
    os.makedirs(parquet_dir, exist_ok=True)

    if dothis:
        start_time = time.time()
        k_start = 0

        maximas = {}
        for i in series_types:
            for k_block in k_blocks:
                if k_block == 0:
                    maximas[f"series{i}_SBM"] = []
                else:
                    maximas[f"series{i}_CBM_k{k_block}"] = []
    
        for k in range(k_start, nbRuns):
            for i in series_types:
                x, t, dt = gen_sample(i, T=6000)
                T = len(x)
                block_max = lambda x: np.max(x, axis=-1)
            
                for k_block in k_blocks:
                    if k_block == 0:
                        maximas[f"series{i}_SBM"].append(nf.sample_blocks(x, T//n_blocks, random=False, circular=True, step = 1, func=block_max))
                    else:
                        r = int(len(x)/n_blocks/k_block)
                        maximas[f"series{i}_CBM_k{k_block}"].append(nf.CBM(x, r, k = k_block, circular=False, step = 1).flatten())
        
            # Save data every saveStep iterations
            if (k + 1) % saveStep == 0:
                for key, data_list in maximas.items():
                    if data_list:  # Only process non-empty lists
                        # Stack all arrays in the list into a single 2D array
                        data_array = np.vstack(data_list)
                        print(f"Saving {key} with shape {data_array.shape}")
                        n_samples, n_features = data_array.shape

                        sample_ids = np.repeat(np.arange(n_samples), n_features)
                        feature_ids = np.tile(np.arange(n_features), n_samples)
                        values = data_array.ravel()

                        df_long = pd.DataFrame({
                            "sample_id": sample_ids,
                            "feature_id": feature_ids,
                            "value": values
                        })
                        ddf = dd.from_pandas(df_long, npartitions=1)

                        parquet_file = os.path.join(parquet_dir, key)

                        if os.path.exists(parquet_file):
                            ddf.to_parquet(parquet_file, append=True, compression="snappy", write_index=False)
                        else:
                            ddf.to_parquet(parquet_file, compression="snappy", write_index=False)
                    
                        # Clear the list after saving
                        maximas[key] = []
            
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
    args = parser.parse_args()

    n_blocks = 10  # Number of blocks
    series_types = [args.series_type] if args.series_type is not None else [0, 1, 2, 3]
    k_blocks = [0, 2, 4]
    print(f"series_types: {series_types}, k_blocks: {k_blocks}")
    gen_block_maxima(n_blocks, series_types=series_types, k_blocks=k_blocks)

if __name__ == "__main__":
    main()