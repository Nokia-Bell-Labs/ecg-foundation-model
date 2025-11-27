import os
import numpy as np
import pandas as pd
import scipy
import torch
from torch.utils.data import Dataset
from scipy.signal import butter, filtfilt
from typing import List, Optional, Callable, Any
from clef.utils.data_augmentation import (
    ecg_positive_augmentation,
)


def is_too_flat(segment: np.ndarray, flat_threshold: float = 1e-3) -> bool:
    return np.var(segment) < flat_threshold


def has_too_many_zeros(segment: np.ndarray, zero_ratio: float = 0.5) -> bool:
    total = np.prod(segment.shape)
    zeros = np.sum(segment == 0)
    return (zeros / total) > zero_ratio


class Normalize:
    def __init__(self, method="z-score"):
        self.method = method

    def apply(self, signal, fs=None):
        if self.method == "z-score":
            mean = np.mean(signal, axis=0, keepdims=True)
            std = np.std(signal, axis=0, keepdims=True) + 1e-8
            return (signal - mean) / std, {"mean": mean, "std": std}
        else:
            raise NotImplementedError
        

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    '''
    Usage: bandpass_filter(stacked_ecg_data, 0.67, 40, fs=500)
    '''
    nyq = 0.5 * fs  # Nyquist Frequency
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")

    filtered_data = np.zeros_like(data)
    for i in range(data.shape[0]):
        filtered_data[i, :] = filtfilt(b, a, data[i, :])

    return filtered_data


class BaseECGContrastiveDataset(Dataset):
    """
    Base dataset for contrastive learning on ECG data.
    Signals from different channels of the same patient are considered positives.
    """

    def __init__(
        self,
        data_path: str,
        metadata_path: Optional[str] = None,
        leads: Optional[List[int]] = None,
        input_len: int = 2500,
        normalization: bool = True,
        do_filter: bool = True,
        transform: Optional[Callable] = None,
        simclr: bool = True,
        use_metadata: bool = True,
        metadata_attributes: Optional[List[str]] = None,
        **kwargs,
    ):
        self.data_path = data_path
        self.leads = leads
        self.input_len = input_len
        self.normalization = normalization
        self.do_filter = do_filter
        self.transform = transform
        self.simclr = simclr
        self.use_metadata = use_metadata
        self.metadata_attributes = metadata_attributes or []

        self.filenames = self._find_files()
        self.headers = {}
        self.samples = []  # List of (fname, channel_idx, channel_name, ch_freq, segments)
        self.patient_to_channels = {}  # patient_id -> list of (ch_idx, ch_name)
        self.patient_to_segments = {}  # patient_id -> segment list

        self.noise_mat = scipy.io.loadmat("dataset/DATA_noises_real.mat")

        for fname in self.filenames:
            header = self._read_header(fname)
            self.headers[fname] = header

            channel_names = header.sig_name
            ch_freq = header.fs if hasattr(header, "fs") else 1

            sig_len = header.sig_len
            segments = []
            for start in range(0, sig_len - input_len + 1, input_len):
                segments.append(start)

            # Extract patient_id from fname
            patient_id = fname.split('/')[-3]
            self.patient_to_channels[patient_id] = [
                (ch_idx, ch_name, ch_freq, fname)
                for ch_idx, ch_name in enumerate(channel_names)
            ]
            self.patient_to_segments[patient_id] = segments

        self.targets = [0 for _ in self.filenames]  # Dummy labels, one per patient

    def _find_files(self) -> List[str]:
        raise NotImplementedError

    def _read_header(self, fname: str) -> Any:
        raise NotImplementedError

    def _read_partial_record(self, fname: str, start: int, end: int) -> np.ndarray:
        raise NotImplementedError
    
    def _read_metadata(self, fname: str) -> pd.DataFrame:
        raise NotImplementedError

    def _get_signal_length(self, header: Any) -> int:
        raise NotImplementedError
    
    def _get_metadata_for_record(self, fname: str) -> Optional[dict]:
        raise NotImplementedError

    def __len__(self):
        # Each file corresponds to one patient
        return len(self.filenames)
    
    def __getitem__(self, idx):
        fname = self.filenames[idx]
        patient_id = fname.split('/')[-3]
        channel_infos = self.patient_to_channels[patient_id]

        if self.use_metadata:
            metadata = self._get_metadata_for_record(fname)
        else:
            metadata = 0

        chid_info1, chid_info2 = np.random.choice(
            len(channel_infos), size=2, replace=True
        )

        ch_info1, ch_info2 = channel_infos[chid_info1], channel_infos[chid_info2]
        ch_idx1, ch_name1, ch_freq1, fname1 = ch_info1
        ch_idx2, ch_name2, ch_freq2, fname2 = ch_info2

        seg1 = self._read_partial_record(fname1, 0, self.input_len)[:, ch_idx1]

        q = 0.8

        if np.random.rand() < q:
            ch_idx2 = ch_idx1
                
        seg2 = self._read_partial_record(fname2, 0, self.input_len)[:, ch_idx2]
        seg1 = self._interpolate_nan(seg1)
        seg2 = self._interpolate_nan(seg2)
        
        seg1 = np.expand_dims(seg1, axis=0)
        seg2 = np.expand_dims(seg2, axis=0)

        if self.do_filter:
            seg1 = bandpass_filter(seg1, 0.67, 40, fs=500)
            seg2 = bandpass_filter(seg2, 0.67, 40, fs=500)

        seg1 = ecg_positive_augmentation(seg1, ch_name1, ch_freq1, self.noise_mat)
        seg2 = ecg_positive_augmentation(seg2, ch_name2, ch_freq2, self.noise_mat)

        if self.normalization:
            norm = Normalize(method="z-score")
            seg1, _ = norm.apply(seg1[0])
            seg2, _ = norm.apply(seg2[0])
            seg1 = np.expand_dims(seg1, axis=0)
            seg2 = np.expand_dims(seg2, axis=0)

        signals = [seg1, seg2]
        labels = [1, 1]  # Both are positives
        ch_names = [ch_name1, ch_name2]
        ch_freqs = [ch_freq1, ch_freq2]

        signals = np.concatenate(signals, axis=0)
        if self.transform:
            signals = self.transform(signals)
        signals = torch.Tensor(signals)
        labels = torch.tensor(labels, dtype=torch.long)

        return signals, labels, ch_names, ch_freqs, metadata

    def _interpolate_nan(self, signal: np.ndarray) -> np.ndarray:
        """
        Interpolate NaN values in the signal using linear interpolation.
        
        Args:
            signal: 1D numpy array that may contain NaN values
            
        Returns:
            signal with NaN values interpolated
        """
        if not np.isnan(signal).any():
            return signal
        
        # Find indices of non-NaN and NaN values
        nan_mask = np.isnan(signal)
        valid_indices = np.where(~nan_mask)[0]
        nan_indices = np.where(nan_mask)[0]
        
        # If all values are NaN, return zeros
        if len(valid_indices) == 0:
            return np.zeros_like(signal)
        
        # If only some values are NaN, interpolate
        if len(nan_indices) > 0:
            signal[nan_indices] = np.interp(
                nan_indices, 
                valid_indices, 
                signal[valid_indices]
            )
        
        return signal

