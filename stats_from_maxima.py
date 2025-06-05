# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import dask.dataframe as dd
import pickle
import os
from dask.distributed import Client
from IPython.display import display

# %%
def init_client():
    # if there’s already a running client, shut it down
    try:
        Client.current().shutdown()
    except (ValueError, RuntimeError):
        pass
    return Client()

def sample_segments_partition(pdf, points_per_series=500, sample_size=100):
    pdf = pdf.reset_index(drop=True)
    pdf['segment_id'] = pdf.index // points_per_series

    def sample_func(group):
        if len(group) >= sample_size:
            return group.sample(n=sample_size, random_state=42)
        else:
            return group  # or use sampling with replacement if needed

    sampled = pdf.groupby('segment_id').apply(sample_func)
    return sampled.drop(columns='segment_id')

def sample_every_n(df, step):
    return df.iloc[0::step]

def agg_within_partition(df, group_size=10):
    # reset local index
    df = df.reset_index(drop=True)
    # group every group_size rows
    grp = df.groupby(df.index // group_size)
    # compute mean and std, suffix columns, then concat
    means = grp.mean().add_suffix('_mean')
    stds  = grp.std(ddof=1).add_suffix('_std')
    return pd.concat([means, stds], axis=1)


def bootstrap_stats_partition(df, group_size=10, n_boot=100):
    """
    Compute bootstrap mean and std for each column in partitioned DataFrame.

    Parameters:
    - df: pandas.DataFrame partition (sampled maxima)
    - group_size: number of rows per bootstrap group
    - n_boot: number of bootstrap resamples

    Returns:
    - pandas.DataFrame with MultiIndex columns [(col, stat)] and row index = group index
    """
    # Ensure clean positional grouping
    df = df.reset_index(drop=True)
    grp_idx = df.index // group_size

    records = []
    for grp, sub in df.groupby(grp_idx):
        stats = {}
        for col in sub.columns:
            vals = sub[col].values
            # bootstrap resamples
            bs_means = np.array([
                np.random.choice(vals, size=len(vals), replace=True).mean()
                for _ in range(n_boot)
            ])
            bs_stds = np.array([
                np.std(np.random.choice(vals, size=len(vals), replace=True), ddof=1)
                for _ in range(n_boot)
            ])
            # average of bootstrap replications
            stats[(col, 'mean')] = bs_means.mean()
            stats[(col, 'std')]  = bs_stds.mean()
        # series named by group index
        rec = pd.Series(stats, name=grp)
        records.append(rec)

    out = pd.DataFrame(records)
    out.index.name = 'group'
    return out

# %%
def main():
    client = init_client()
    display(client)

    data_folder = "./maximas_data"
    # data is structured as: f"{name}-{stamp}
    # Get all files in the data folder
    files = os.listdir(data_folder)

    # Extract names and stamps from filenames
    names = set()
    stamps = set()

    for file in files:
        if '-' in file:
            parts = file.split('-')
            if len(parts) >= 2:
                name = parts[0]
                stamp = parts[1].split('.')[0]  # Remove file extension if present
                names.add(name)
                stamps.add(stamp)

    print(f"Unique names: {sorted(names)}")
    print(f"Unique stamps: {sorted(stamps)}")



    # %%
    maximas_data = {}
    for name in names:
        maximas_data[name] = []
        for stamp in stamps:
            file_path = os.path.join(data_folder, f"{name}-{stamp}.parquet")
            if os.path.exists(file_path):
                ddf = dd.read_parquet(file_path)
                maximas_data[name].append(ddf)
        maximas_data[name] = dd.concat(maximas_data[name], axis=0)

    # %%
    # maximas_data[name].partitions[0:1].memory_usage(deep=True).sum().compute() / 100e6

    # %%
    points_per_series = 655360
    points_per_block = points_per_series // 10

    disjoint_maximas = {}

    for name in maximas_data.keys():
        sampled = maximas_data[name].map_partitions(sample_every_n, step=points_per_block)
        disjoint_maximas[name] = sampled


    # %%
    overlapping_maximas = {}
    for name in maximas_data.keys():
        sampled = maximas_data[name]["SBM"].map_partitions(sample_every_n, step=points_per_block // 10)
        overlapping_maximas[name] = sampled

    # %%
    sampled_maximas = {}
    for name in maximas_data.keys():
        sampled_maximas[name] = maximas_data[name]["SBM"].map_partitions(
            sample_segments_partition,
            points_per_series=points_per_series,
            sample_size=100,
            meta = maximas_data[name]._meta
        )

    # %%
    # Dictionary to store mean estimators
    standard_disjoint_estimator = {}
    # Dictionary to store standard deviation estimators
    standard_disjoint_std_estimator = {}

    for name, ddf in disjoint_maximas.items():
        # build a minimal “meta” so Dask knows the output dtypes/columns
        cols      = ddf.columns
        meta_cols = [f"{c}_mean" for c in cols] + [f"{c}_std" for c in cols]
        meta      = pd.DataFrame(columns=meta_cols, dtype=float)

        # compute one DF with all _mean and _std columns
        combined = (
            ddf
            .map_partitions(agg_within_partition, group_size=10, meta=meta)
            .compute()
        )

        # split into two DataFrames
        means = combined[[c for c in combined.columns if c.endswith('_mean')]]
        stds  = combined[[c for c in combined.columns if c.endswith('_std')]]

        # store in your two dicts
        standard_disjoint_estimator[name]     = means
        standard_disjoint_std_estimator[name] = stds

    print("Standard disjoint estimators computed.")

    # %%
    # === Applying in Dask ===
    # Assume `disjoint_maximas` is your dict of Dask DataFrames
    bs_disjoint_estimator     = {}
    bs_disjoint_std_estimator = {}

    for name, ddf in disjoint_maximas.items():
        # prepare meta with MultiIndex columns [(col, 'mean'), (col, 'std')]
        cols = ddf.columns
        mi = pd.MultiIndex.from_product([cols, ['mean', 'std']])
        meta = pd.DataFrame(columns=mi, dtype=float)

        # map partitions to compute mean & std together
        bs_stats_dd = ddf.map_partitions(
            bootstrap_stats_partition,
            group_size=10,
            n_boot=100,
            meta=meta
        )

        # compute the combined stats DataFrame
        bs_stats = bs_stats_dd.compute()

        # split into two DataFrames: means and stds
        means = bs_stats.xs('mean', axis=1, level=1).sort_index()
        stds  = bs_stats.xs('std',  axis=1, level=1).sort_index()

        bs_disjoint_estimator[name]     = means
        bs_disjoint_std_estimator[name] = stds

    print("Bootstrap disjoint estimators computed.")
    # Now `bs_disjoint_estimator` and `bs_disjoint_std_estimator` hold your results per series.

    # %%
    # Bundle both dicts into one object
    estimators = {
        'means': {
            "standard_disjoint_estimator": standard_disjoint_estimator,
            "bs_disjoint_estimator": bs_disjoint_estimator
        },
        'stds': {
            "standard_disjoint_estimator": standard_disjoint_std_estimator,
            "bs_disjoint_estimator": bs_disjoint_std_estimator
        },
    }

    # Write to disk
    with open('./estimators.pkl', 'wb') as f:
        pickle.dump(estimators, f)

    print("Estimators saved to disk.")

    # %%
    # 1) Compute summary stats for each series
    summary_stats = {}
    for name, ddf in disjoint_maximas.items():
        # ddf.describe() returns a small pandas DataFrame once you compute()
        summary_stats[name] = (
            ddf
            .describe()                   # count, mean, std, min, 25/50/75%, max
            .compute()                    
        )

    # 2) Pickle the result
    with open('disjoint_maximas_summary.pkl', 'wb') as f:
        pickle.dump(summary_stats, f)

    client.shutdown()
    print("Summary stats for disjoint maximas saved to disk.")
    print("Completed")

    # %%

if __name__ == "__main__":
    main()