"""
create_subsets.py

Description:
    This script builds the NumPy subsets files (train, validation, test) needed
    by metaqnn.

Authors : 
    Giuseppe Chiari (giuseppe.chiari@polimi.it),
    Davide Galli (davide.galli@polimi.it),
    Davide Zoni (davide.zoni@polimi.it)
"""


import os
import argparse
import h5py
import scipy.signal
import numpy as np

from npy_append_array import NpyAppendArray
from tqdm.auto import tqdm

from matplotlib import pyplot as plt

# Seed
np.random.seed(1234)

# Define total length of a single trace
BATCH_SIZE = 134_217_550

# AES S-Box
aes_sbox = np.array(
    (0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16), dtype=np.uint8)


def highpass(traces: np.ndarray,
             Wn: float = 0.002) -> np.ndarray:
    """
    Applies a 3rd-order high-pass filter to the traces.

    Parameters
    ----------
    `traces`: array_like
        The traces to filter.
    `Wn`    : float, optional
        The cutoff frequency of the filter, express relative to the Nyquist frequency (default is 0.002).

    Returns
    ----------
    The filtered traces.
    """

    b, a = scipy.signal.butter(3, Wn)
    y = scipy.signal.filtfilt(b, a, traces).astype(np.float32)

    return (traces - y).astype(np.float32)


def aggregate(trace: np.ndarray,
              n: int) -> np.ndarray:
    """
    Aggregates `n` consective sample of the trace.

    Parameters
    ----------
    `trace` : array_like
        The trace to aggregate.
    `n`     : int
        The number of sample to aggregate.

    Returns
    ----------
    The aggregated trace.
    """

    # Divide array into n subsections
    num_chunks = int(trace.shape[1] / n)
    chunks = np.array_split(trace, num_chunks, axis=1)
    # Compute the mean of each subsection
    means = [np.mean(chunk, axis=1, dtype=np.float32) for chunk in chunks]

    return np.array(means, dtype=np.float32).T


def norm(traces: np.ndarray,
         axis = 0) -> np.ndarray:
    r"""
    Normalizes the traces.

    .. math::
        x = \\frac{x - \\mu}{\\sigma}

    Parameters
    ----------
    `traces` : array_like
        The traces to normalize.
    `axis`   : int, optional
        The axis along which the normalization is performed (default is 0).

    Returns
    ----------
    The normalized traces.
    """

    return (traces - np.mean(traces, axis=axis, keepdims=True)) / np.std(traces, axis=axis, keepdims=True)


def _computeTargets(plains, key):
    return np.array([aes_sbox[plains[:, byte] ^ key[byte]] for byte in range(16)]).T


def _writeConfig(dataset_folder: str,
                 train_shape,
                 valid_shape,
                 test_shape,
                 window,
                 aggregate_n_samples: int):

    config_path = os.path.join(dataset_folder, "config.txt")
    with open(config_path, "w") as file:
        file.write(f"N. training traces: {train_shape[0]}\n")
        file.write(f"N. validation traces: {valid_shape[0]}\n")
        file.write(f"N. test traces: {test_shape[0]}\n")
        file.write(f"N. input samples: {train_shape[1]}\n")
        file.write(f"Window: {window}\n")
        file.write(f"Aggregation: {aggregate_n_samples}\n")


def plotTargetStatistics(dataset_folder: str) -> None:
    """
    Plot the distribution of the targets in the dataset and save the plot in `dataset_folder`.
    The three datasets should have the same distribution.

    Parameters
    ----------
    `dataset_folder` : str
        The folder where the dataset is stored.
    """
    train_tar_path = os.path.join(dataset_folder, 'train_targets.npy')
    valid_tar_path = os.path.join(dataset_folder, 'valid_targets.npy')
    test_tar_path = os.path.join(dataset_folder, 'test_targets.npy')

    train_targets = np.load(train_tar_path, mmap_mode='r')
    train_stats = np.unique(train_targets, return_counts=True)
    valid_targets = np.load(valid_tar_path, mmap_mode='r')
    valid_stats = np.unique(valid_targets, return_counts=True)
    test_targets = np.load(test_tar_path, mmap_mode='r')
    test_stats = np.unique(test_targets, return_counts=True)

    plt.figure(figsize=(9, 4))
    ax = plt.subplot(1, 3, 1)
    ax.hist(norm(train_stats[1]), bins=int(180/5))
    ax.set_title("Train")

    ax = plt.subplot(1, 3, 2)
    ax.hist(norm(valid_stats[1]), bins=int(180/5))
    ax.set_title("Validation")

    ax = plt.subplot(1, 3, 3)
    ax.hist(norm(test_stats[1]), bins=int(180/5))
    ax.set_title("Test")

    plt.savefig(os.path.join(dataset_folder,
                'target_statistics.png'), bbox_inches='tight')
    plt.show()
    plt.close()


def printDatasetInfo(dataset_folder: str):
    """
    Print dataset information.

    Parameters
    ----------
    `dataset_folder` : str
        The folder where the dataset is stored.
    """
    config_file = os.path.join(dataset_folder, "config.txt")
    with open(config_file, "r") as file:
        info = file.read()
    print(info)


def cutTracesIntoExecutions(trace, starts, window):
    
    aligned_execs = []

    for start in starts:
        aligned_execs.append(trace[start:start+window[1]])
    
    return np.asarray(aligned_execs)


def preprocess(traces, window, aggregate_n_sample: int):
    traces = traces[:, window[0]:window[1]]
    if aggregate_n_sample > 1:
        traces = aggregate(traces, aggregate_n_sample)
    return traces


def getSubsetsSplits(chunk_files, window, split_traces: float):

    # Look for minimum number of executions in a single trace
    min_execs = -1

    for chunk_file in chunk_files:
        h5_file = h5py.File(chunk_data_dir+chunk_file, 'r')
        h5_pinpoints = h5_file['metadata/pinpoints']

        keys = np.sort(list(h5_pinpoints.keys()))
        key_ids = [k.split('_')[1] for k in keys]

        # Iterate over key identifiers
        for key_id in key_ids:

            tmp_min_execs = len([start for start in h5_pinpoints['pinpoints_' + key_id]['start'] 
                                    if start+(window[1] - window[0]) < BATCH_SIZE])
            if tmp_min_execs < min_execs or min_execs == -1:
                min_execs = tmp_min_execs
        
    # Create subsets splits
    n_train_traces = round(min_execs * split_traces)
    n_valid_traces = round(min_execs * (1-split_traces) / 2)
    n_test_traces = min_execs - n_train_traces - n_valid_traces

    return n_train_traces, n_valid_traces, n_test_traces


if __name__ == '__main__':
    '''
    python create_subsets.py --chunk_data_dir <"/path/to/.h5/files/">
                             --out_data_dir <"/path/to/numpy/subsets/output/folder/">
                             --split_traces <TRAIN_SPLIT_PERCENTAGE>
                             --window <START_SAMPLE END_SAMPLE>
                             --aggregate <AGGREGATION_TERM>
    '''

    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("-dd", "--chunk_data_dir", required=True, help="Path to the folder containing the .h5 files chunks. No default.")
    parser.add_argument("-od", "--out_data_dir",  required=True, help="Path to the output folder where subsets will be stored. No default.")
    parser.add_argument("-st", "--split_traces",  required=False, default=0.8, help="Subsets split percentage, \
                        only the train split is to be specified, validation and test splits are inferred. Default 0.8.")
    parser.add_argument("-w",  "--window",
                        required=False, default=[0, BATCH_SIZE], nargs='+', help=f"Samples window to extract from each execution. Default is the entire trace 0 {BATCH_SIZE}.")
    parser.add_argument("-a",  "--aggregate",     required=False, default=1, help="Aggregation term. Default is 1.")

    args = vars(parser.parse_args())

    chunk_data_dir = args['chunk_data_dir']
    out_data_dir = args['out_data_dir']
    split_traces = float(args['split_traces'])
    window = [int(i) for i in args['window']]
    aggregate_n_sample = int(args['aggregate'])

    if not os.path.exists(out_data_dir):
        os.makedirs(out_data_dir)


    chunk_files = np.sort(next(os.walk(chunk_data_dir))[2])

    train_win_path = os.path.join(out_data_dir, 'train_windows.npy')
    train_tar_path = os.path.join(out_data_dir, 'train_targets.npy')
    train_meta_path = os.path.join(out_data_dir, 'train_meta.npy')
    valid_win_path = os.path.join(out_data_dir, 'valid_windows.npy')
    valid_tar_path = os.path.join(out_data_dir, 'valid_targets.npy')
    valid_meta_path = os.path.join(out_data_dir, 'valid_meta.npy')
    test_win_path = os.path.join(out_data_dir, 'test_windows.npy')
    test_tar_path = os.path.join(out_data_dir, 'test_targets.npy')
    test_meta_path = os.path.join(out_data_dir, 'test_meta.npy')

    n_train_traces, n_valid_traces, n_test_traces = getSubsetsSplits(chunk_files, window, split_traces)

    with NpyAppendArray(train_win_path, delete_if_exists=True) as npa_train_win, \
            NpyAppendArray(train_tar_path, delete_if_exists=True) as npa_train_tar,  \
            NpyAppendArray(train_meta_path, delete_if_exists=True) as npa_train_meta, \
            NpyAppendArray(valid_win_path, delete_if_exists=True) as npa_valid_win,  \
            NpyAppendArray(valid_tar_path, delete_if_exists=True) as npa_valid_tar,  \
            NpyAppendArray(valid_meta_path, delete_if_exists=True) as npa_valid_meta, \
            NpyAppendArray(test_win_path, delete_if_exists=True) as npa_test_win,    \
            NpyAppendArray(test_tar_path, delete_if_exists=True) as npa_test_tar,    \
            NpyAppendArray(test_meta_path, delete_if_exists=True) as npa_test_meta:


        for chunk_file in tqdm(chunk_files, total=len(chunk_files), desc="Creating attack subsets", position=0):

            # Open .h5 chunk file
            h5_file = h5py.File(chunk_data_dir+chunk_file, 'r')

            # Get key identifiers
            h5_traces = h5_file['data/traces']
            keys = np.sort(list(h5_traces.keys()))
            key_ids = [k.split('_')[1] for k in keys]

            # Iterate over key identifiers
            for key_id in tqdm(key_ids, desc=f'Iterating traces in {chunk_file}', position=1, leave=False):

                # Read i-th trace
                trace = h5_traces['trace_' + key_id][:].astype(np.float32)

                # Read i-th key 'k'
                h5_ciphers = h5_file['metadata/ciphers']
                key = h5_ciphers['ciphers_' + key_id + '/key']['k'][0]

                # Read i-th pinpoints: 'start' samples of each cipher execution
                h5_pinpoints = h5_file['metadata/pinpoints']
                starts = [start for start in h5_pinpoints['pinpoints_' + key_id]['start'] if start+(window[1] - window[0]) < BATCH_SIZE]
                
                # Cut i-th trace into batches
                aligned_execs = cutTracesIntoExecutions(highpass(trace), starts, window)
                
                # Read i-th plaintexts 'p'
                plaintexts = np.asarray(h5_ciphers['ciphers_' + key_id + '/plaintexts']['p'][:len(aligned_execs)])
                
                # Get number of executions
                n_traces = len(aligned_execs)
                
                assert n_traces == len(plaintexts), \
                        f"Number of traces {n_traces} and plains {len(plaintexts)} must be the same"

                # Compute targets bytes
                targets = _computeTargets(plaintexts, key)

                # Randomize dataset
                indices = np.arange(len(aligned_execs))
                np.random.shuffle(indices)

                aligned_execs = aligned_execs[indices]
                targets = targets[indices]
                plaintexts = plaintexts[indices]

                # Preprocess executions: aggregate them
                aligned_execs = preprocess(
                    aligned_execs, window, aggregate_n_sample)

                ### Build subsets 
                # Training
                execs = aligned_execs[:n_train_traces]
                trgts = targets[:n_train_traces]
                plains = plaintexts[:n_train_traces]

                npa_train_win.append(execs)
                npa_train_tar.append(trgts)
                npa_train_meta.append(
                    np.stack((plains, np.broadcast_to(key, (n_train_traces, 16))), axis=1))

                # Validation
                execs = aligned_execs[n_train_traces:n_train_traces+n_valid_traces]
                trgts = targets[n_train_traces:n_train_traces+n_valid_traces]
                plains = plaintexts[n_train_traces:n_train_traces+n_valid_traces]

                npa_valid_win.append(execs)
                npa_valid_tar.append(trgts)
                npa_valid_meta.append(
                    np.stack((plains, np.broadcast_to(key, (n_valid_traces, 16))), axis=1))

                # Test
                execs = aligned_execs[-n_test_traces:]
                trgts = targets[-n_test_traces:]
                plains = plaintexts[-n_test_traces:]

                npa_test_win.append(execs)
                npa_test_tar.append(trgts)
                npa_test_meta.append(
                    np.stack((plains, np.broadcast_to(key, (n_test_traces, 16))), axis=1))
                
            h5_file.close()


    # Write configuration .txt file
    _writeConfig(out_data_dir,
                 np.load(train_win_path, mmap_mode='r').shape,
                 np.load(valid_win_path, mmap_mode='r').shape,
                 np.load(test_win_path, mmap_mode='r').shape,
                 window,
                 aggregate_n_sample)

    # Plot targets statistics and print dataset info - Optional
    plotTargetStatistics(out_data_dir)
    printDatasetInfo(out_data_dir)
