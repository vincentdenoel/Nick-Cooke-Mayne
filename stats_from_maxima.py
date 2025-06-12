# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import dask.dataframe as dd
import pickle
import os

data_folder = "./maximas_data_sameSuperblock"
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

def sample_every_n(df, step):
    return df.iloc[0::step]

for name in maximas_data.keys():
    disjoint_maximas[name] = maximas_data[name][["SBM"]].map_partitions(sample_every_n, step=points_per_block)


# %%
overlapping_maximas = {}
for name in maximas_data.keys():
     overlapping_maximas[name] = maximas_data[name].map_partitions(sample_every_n, step=points_per_block // 10)

# %%
def sample_segments_partition(pdf, points_per_series=500, sample_size=100):
    pdf = pdf.reset_index(drop=True)
    pdf['segment_id'] = pdf.index // points_per_series

    def sample_func(group):
        if len(group) >= sample_size:
            return group.sample(n=sample_size, random_state=42)
        else:
            return group.sample(n=sample_size, replace=True, random_state=42)  # Sampling with replacement

    sampled = pdf.groupby('segment_id').apply(sample_func, include_groups=False)
    return sampled

sampled_maximas = {}
for name in maximas_data.keys():
    sampled_maximas[name] = maximas_data[name][["SBM"]].map_partitions(
        sample_segments_partition,
        points_per_series=points_per_series,
        sample_size=100,
        meta = maximas_data[name][["SBM"]]._meta
    )

print("Lazy maxima dicts created")

# %%
def agg_within_partition(df, group_size=10):
    # reset local index
    df = df.reset_index(drop=True)
    # group every group_size rows
    grp = df.groupby(df.index // group_size)
    # compute mean and std, suffix columns, then concat
    means = grp.mean().add_suffix('_mean')
    stds  = grp.std(ddof=1).add_suffix('_std')
    return pd.concat([means, stds], axis=1)

# Function to apply standard aggregation to any maxima type
def apply_standard_agg(maximas_dict, group_size):
    estimator = {}
    std_estimator = {}
    
    for name, ddf in maximas_dict.items():
        # build a minimal "meta" so Dask knows the output dtypes/columns
        cols = ddf.columns
        meta_cols = [f"{c}_mean" for c in cols] + [f"{c}_std" for c in cols]
        meta = pd.DataFrame(columns=meta_cols, dtype=float)

        # compute one DF with all _mean and _std columns
        combined = (
            ddf
            .map_partitions(agg_within_partition, group_size=group_size, meta=meta)
            .compute()
        )

        # split into two DataFrames
        means = combined[[c for c in combined.columns if c.endswith('_mean')]]
        stds = combined[[c for c in combined.columns if c.endswith('_std')]]

        # store in your two dicts
        estimator[name] = means
        std_estimator[name] = stds
    
    return estimator, std_estimator

# Apply standard aggregation to different maxima types
standard_disjoint_estimator, standard_disjoint_std_estimator = apply_standard_agg(disjoint_maximas, 10)
print("Standard disjoint estimators computed.")

standard_overlapping_estimator, standard_overlapping_std_estimator = apply_standard_agg(overlapping_maximas, 100)
print("Standard overlapping estimators computed.")

standard_sampled_estimator, standard_sampled_std_estimator = apply_standard_agg(sampled_maximas, 100)
print("Standard sampled estimators computed.")

# %%
def bootstrap_stats_partition(df, group_size=10, n_boot=1000, block_size=1):
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
            if col == 'SBM': # Handle SBM where block_size is always 1
                group_size *= block_size
                block_size == 1
            
            vals = sub[col].values
    
            # Reshape vals into a 2D array with block_size column
            n_rows = len(vals) // block_size
    
            # Truncate vals to fit evenly into blocks
            vals_2d = vals[:n_rows * block_size].reshape(n_rows, block_size)
    
            # bootstrap resamples with block sampling
            bs_means = np.array([
                # Sample entire rows (blocks) with replacement
                vals_2d[np.random.choice(n_rows, size=n_rows, replace=True)].flatten().mean()
                for _ in range(n_boot)
            ])
            bs_stds = np.array([
                # Sample entire rows (blocks) with replacement
                np.std(vals_2d[np.random.choice(n_rows, size=n_rows, replace=True)].flatten(), ddof=1)
                for _ in range(n_boot)
            ])
    
            # average of bootstrap replications
            stats[(col, 'mean')] = bs_means.mean()
            stats[(col, 'std')] = bs_stds.mean()
        # series named by group index
        rec = pd.Series(stats, name=grp)
        records.append(rec)

    out = pd.DataFrame(records)
    out.index.name = 'group'
    return out

# Function to apply bootstrap aggregation to any maxima type
def apply_bootstrap_agg(maximas_dict, group_size=10, n_boot=1000, block_size=1):
    bs_estimator = {}
    bs_std_estimator = {}
    
    for name, ddf in maximas_dict.items():
        # prepare meta with MultiIndex columns [(col, 'mean'), (col, 'std')]
        cols = ddf.columns
        mi = pd.MultiIndex.from_product([cols, ['mean', 'std']])
        meta = pd.DataFrame(columns=mi, dtype=float)

        # map partitions to compute mean & std together
        bs_stats_dd = ddf.map_partitions(
            bootstrap_stats_partition,
            group_size=group_size,
            n_boot=n_boot,
            meta=meta
        )

        # compute the combined stats DataFrame
        bs_stats = bs_stats_dd.compute()

        # split into two DataFrames: means and stds
        means = bs_stats.xs('mean', axis=1, level=1).sort_index()
        stds = bs_stats.xs('std', axis=1, level=1).sort_index()

        bs_estimator[name] = means
        bs_std_estimator[name] = stds
    
    return bs_estimator, bs_std_estimator

# Apply bootstrap aggregation to different maxima types
bs_sampled_estimator, bs_sampled_std_estimator = apply_bootstrap_agg(sampled_maximas, group_size=100)
print("Bootstrap sampled estimators computed.")

bs_overlapping_estimator, bs_overlapping_std_estimator = apply_bootstrap_agg(overlapping_maximas, group_size=100, n_boot = 100, block_size=10)
print("Bootstrap overlapping estimators computed.")

bs_disjoint_estimator, bs_disjoint_std_estimator = apply_bootstrap_agg(disjoint_maximas, group_size=10)
print("Bootstrap disjoint estimators computed.")

# %%
# Bundle all dicts into one object
estimators = {
    'means': {
        "standard_disjoint_estimator": standard_disjoint_estimator,
        "bs_disjoint_estimator": bs_disjoint_estimator,
        "standard_overlapping_estimator": standard_overlapping_estimator,
        "bs_overlapping_estimator": bs_overlapping_estimator,
        "standard_sampled_estimator": standard_sampled_estimator,
        "bs_sampled_estimator": bs_sampled_estimator
    },
    'stds': {
        "standard_disjoint_std_estimator": standard_disjoint_std_estimator,
        "bs_disjoint_std_estimator": bs_disjoint_std_estimator,
        "standard_overlapping_std_estimator": standard_overlapping_std_estimator,
        "bs_overlapping_std_estimator": bs_overlapping_std_estimator,
        "standard_sampled_std_estimator": standard_sampled_std_estimator,
        "bs_sampled_std_estimator": bs_sampled_std_estimator
    },
}

# Write to disk
with open('./estimators.pkl', 'wb') as f:
    pickle.dump(estimators, f)

print("Estimators saved to disk.")

# %%
# Function to compute and store summary stats for a type of maxima
def compute_summary_stats(maximas_by_type):
    summary_stats = {}
    for type_name, maximas_dict in maximas_by_type.items():
        summary_stats[type_name] = {}
        for name, ddf in maximas_dict.items():
            summary_stats[type_name][name] = ddf.describe().compute()
    return summary_stats

# Organize maximas by type
maximas_by_type = {
    'disjoint': disjoint_maximas,
    'overlapping': overlapping_maximas,
    'sampled': sampled_maximas
}

# Compute summary stats for all types
summary_stats = compute_summary_stats(maximas_by_type)

# Pickle the result
with open('all_maximas_summary.pkl', 'wb') as f:
    pickle.dump(summary_stats, f)

print("Summary stats for all maximas types saved to disk.")
print("Completed")

# %%
