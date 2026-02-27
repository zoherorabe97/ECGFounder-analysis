import numpy as np
import pandas as pd
import wfdb
from scipy import signal
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from scipy.signal import medfilt, iirnotch, filtfilt, butter, resample
from scipy.interpolate import interp1d
import scipy.io
from util import filter_bandpass


import h5py


class LVEF_12lead_cls_Dataset(Dataset):
    def __init__(self, ecg_path, labels_df, target_fs=5000, transform=None):
        """
        Args:
            labels_df (DataFrame): DataFrame containing columns:
                                   [path, ?, ?, fs, label_0, label_1, ..., label_N]
                                   Position 0: path to .mat file
                                   Position 3: sampling frequency
                                   Position 4+: label values
            ecg_path (str, optional): Base ECG path (deprecated, kept for compatibility)
            target_fs (int): Target sampling frequency for resampling (fixed output length)
            transform (callable, optional): Optional transform to be applied
        """
        self.labels_df = labels_df.reset_index(drop=True)
        self.ecg_path = ecg_path
        self.target_fs = target_fs
        self.transform = transform
        
        # Lead configuration for reordering if needed
        self.input_leads = ['I', 'II', 'III', 'aVR', 'aVF', 'aVL', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        self.new_leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        self.lead_indices = [self.input_leads.index(lead) for lead in self.new_leads]
        self.expected_leads = 12

    # =========================================================
    # LOAD DATA FROM .MAT OR .H5 FILE
    # =========================================================
    def load_data(self, file_path):
        """
        Load a MATLAB file (.mat) or HDF5 file (.h5/.hdf5) containing ECG data
        
        Args:
            file_path (str): path to .mat or .h5/.hdf5 file
            
        Returns:
            signal: ECG signal as numpy array (leads, time)
        """
        if file_path.endswith('.h5') or file_path.endswith('.hdf5'):
            return self._load_h5(file_path)
        elif file_path.endswith('.mat'):
            return self._load_mat(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}. Expected .mat, .h5, or .hdf5 file")

    def _load_mat(self, file_path):
        """
        Load MATLAB (.mat) file
        
        Args:
            file_path (str): path to .mat file
            
        Returns:
            signal: ECG signal as numpy array
        """
        try:
            mat_data = scipy.io.loadmat(file_path)
            # Try common key names for ECG data
            if 'val' in mat_data:
                signal = np.asarray(mat_data['val'], dtype=np.float64)
            elif 'ecg' in mat_data:
                signal = np.asarray(mat_data['ecg'], dtype=np.float64)
            elif 'signal' in mat_data:
                signal = np.asarray(mat_data['signal'], dtype=np.float64)
            else:
                # If none of the common keys exist, try the first valid key
                # (MATLAB adds '__version__', '__header__', '__globals__' as metadata)
                valid_keys = [k for k in mat_data.keys() if not k.startswith('__')]
                if valid_keys:
                    signal = np.asarray(mat_data[valid_keys[0]], dtype=np.float64)
                else:
                    raise ValueError(f"No valid data keys found in {file_path}")
            return signal
        except Exception as e:
            raise RuntimeError(f"Error reading .mat file {file_path}: {e}")

    def _load_h5(self, file_path):
        """
        Load HDF5 (.h5 or .hdf5) file
        
        Args:
            file_path (str): path to .h5 or .hdf5 file
            
        Returns:
            signal: ECG signal as numpy array
        """
        try:
            with h5py.File(file_path, 'r') as f:
                # Try common key names for ECG data
                if 'ecg' in f:
                    signal = f['ecg'][()]
                elif 'val' in f:
                    signal = f['val'][()]
                elif 'signal' in f:
                    signal = f['signal'][()]
                else:
                    # If none of the common keys exist, try the first dataset
                    key = list(f.keys())[0]
                    signal = f[key][()]
            return np.asarray(signal, dtype=np.float64)
        except Exception as e:
            raise RuntimeError(f"Error reading .h5 file {file_path}: {e}")

    # =========================================================
    # ENSURE CORRECT SHAPE
    # =========================================================
    def ensure_correct_shape(self, signal_data, expected_leads=None):
        """
        Ensure signal has shape (leads, time)
        
        Args:
            signal_data (ndarray): ECG signal
            expected_leads (int): expected number of leads (default 12)
            
        Returns:
            signal_data: signal with correct shape (leads, time)
        """
        if expected_leads is None:
            expected_leads = self.expected_leads
        
        # Handle case where shape is (time, leads) instead of (leads, time)
        if signal_data.shape[0] != expected_leads and signal_data.shape[1] == expected_leads:
            signal_data = signal_data.T
        
        # Validate shape
        if signal_data.shape[0] != expected_leads:
            raise ValueError(
                f"Expected {expected_leads} leads, got {signal_data.shape[0]}. "
                f"Signal shape: {signal_data.shape}"
            )
        
        return signal_data

    # =========================================================
    # HANDLE NAN AND EXTREME VALUES
    # =========================================================
    def handle_invalid_values(self, signal_data, max_safe_value=1e6):
        """
        Handle NaN, Inf, and extreme values in signal
        
        Args:
            signal_data (ndarray): ECG signal
            max_safe_value (float): clipping threshold
            
        Returns:
            signal_data: cleaned signal
        """
        # Replace NaN and Inf with 0
        signal_data = np.nan_to_num(signal_data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Clip extreme values
        signal_data = np.clip(signal_data, -max_safe_value, max_safe_value)
        
        return signal_data

    # =========================================================
    # REORDER LEADS
    # =========================================================
    def reorder_leads(self, signal_data):
        """
        Reorder leads if necessary using lead_indices
        
        Args:
            signal_data (ndarray): ECG signal (12, time)
            
        Returns:
            signal_data: reordered signal
        """
        if len(self.lead_indices) == 12:
            signal_data = signal_data[self.lead_indices, :]
        
        return signal_data

    # =========================================================
    # NORMALIZATION
    # =========================================================
    def z_score_normalization(self, signal):
        """
        Apply global z-score normalization across all leads
        
        Args:
            signal (ndarray): ECG signal (leads, time)
            
        Returns:
            normalized signal
        """
        mean = np.mean(signal)
        std = np.std(signal)
        return (signal - mean) / (std + 1e-8)

    # =========================================================
    # RESAMPLING
    # =========================================================
    def resample_unequal(self, ts, fs_in, fs_out):
        """
        Resample signal from fs_in to fs_out (fixed output length)
        
        Args:
            ts (ndarray): input signal of shape (n_leads, n_samples)
            fs_in (int): input sampling frequency
            fs_out (int): output sampling frequency (or fixed output length)
        
        Returns:
            resampled signal of shape (n_leads, fs_out)
        """
        if fs_in == 0 or len(ts) == 0:
            return ts

        duration = ts.shape[1] / fs_in
        fs_in, fs_out = int(fs_in), int(fs_out)

        # Early exit if no resampling needed
        if fs_out == fs_in:
            return ts
        if 2 * fs_out == fs_in:
            return ts[:, ::2]

        # Create output array with fixed size
        resampled_ts = np.zeros((ts.shape[0], fs_out))
        
        # Create interpolation points
        x_old = np.linspace(0, duration, num=ts.shape[1], endpoint=True)
        x_new = np.linspace(0, duration, num=fs_out, endpoint=True)

        # Interpolate each lead
        for i in range(ts.shape[0]):
            f = interp1d(x_old, ts[i, :], kind='linear')
            resampled_ts[i, :] = f(x_new)

        return resampled_ts

    # =========================================================
    # FINAL VALIDATION
    # =========================================================
    def validate_final_signal(self, signal_data):
        """
        Final safety check for NaN/Inf values
        
        Args:
            signal_data (ndarray): ECG signal
            
        Returns:
            signal_data: validated signal
        """
        if not np.isfinite(signal_data).all():
            # Log warning and replace with zeros
            print(f"Warning: Non-finite values detected in signal. Replacing with zeros.")
            signal_data = np.zeros_like(signal_data)
        
        return signal_data

    # =========================================================
    # DATASET LENGTH
    # =========================================================
    def __len__(self):
        """Return number of samples in dataset"""
        return len(self.labels_df)

    # =========================================================
    # MAIN DATA LOADING
    # =========================================================
    def __getitem__(self, idx):
        """
        Load and process a single sample
        
        DataFrame structure expected:
        Position 0: path (str) - path to .mat file
        Position 1: column 1 (unused)
        Position 2: column 2 (unused)
        Position 3: fs (int) - sampling frequency
        Position 4+: label values (float) - label vector
        
        Returns:
            signal_tensor: torch.Tensor of shape (12, target_fs)
            labels_tensor: torch.Tensor of label vector
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.labels_df.iloc[idx]

        # -------------------------------------------------------
        # 1. Extract metadata from row using positional indexing
        # -------------------------------------------------------
        file_path = str(row.iloc[0])  # path (column 0)
        fs_in = int(row.iloc[3])      # fs (column 3)
        labels = row.iloc[4:].values.astype(np.float32)  # columns from index 4 onwards

        # -------------------------------------------------------
        # 2. Load .mat file
        # -------------------------------------------------------
        try:
            signal_data = self.load_data(file_path)
        except Exception as e:
            raise RuntimeError(f"Error loading {file_path}: {e}")

        # -------------------------------------------------------
        # 3. Ensure correct shape (leads, time)
        # -------------------------------------------------------
        try:
            signal_data = self.ensure_correct_shape(signal_data)
        except Exception as e:
            raise RuntimeError(
                f"Error processing shape for {file_path}: {str(e)}. "
                f"Got shape {signal_data.shape}"
            )

        # -------------------------------------------------------
        # 4. Ensure float32 type
        # -------------------------------------------------------
        signal_data = signal_data.astype(np.float32)

        # -------------------------------------------------------
        # 5. Handle NaNs and extreme values
        # -------------------------------------------------------
        signal_data = self.handle_invalid_values(signal_data)

        # -------------------------------------------------------
        # 6. Reorder leads if necessary
        # -------------------------------------------------------
        signal_data = self.reorder_leads(signal_data)

        # -------------------------------------------------------
        # 7. Apply z-score normalization
        # -------------------------------------------------------
        signal_data = self.z_score_normalization(signal_data)

        # -------------------------------------------------------
        # 8. Resample to target_fs (fixed output length)
        # -------------------------------------------------------
        signal_data = self.resample_unequal(signal_data, fs_in, self.target_fs)

        # -------------------------------------------------------
        # 9. Final safety check for NaNs/Infs after resampling
        # -------------------------------------------------------
        signal_data = self.handle_invalid_values(signal_data)

        # -------------------------------------------------------
        # 10. Final validation
        # -------------------------------------------------------
        signal_data = self.validate_final_signal(signal_data)

        # -------------------------------------------------------
        # 11. Convert to torch tensors
        # -------------------------------------------------------
        signal_tensor = torch.tensor(signal_data, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.float32)

        # -------------------------------------------------------
        # 12. Optional transform
        # -------------------------------------------------------
        if self.transform:
            signal_tensor = self.transform(signal_tensor)

        return signal_tensor, labels_tensor
'''
class LVEF_12lead_cls_Dataset(Dataset):
    def __init__(self, ecg_path , labels_df, target_fs=5000, transform=None):
        """
        Args:
            labels_df (DataFrame): DataFrame containing columns:
                                   [path, ?, ?, fs, label_0, label_1, ..., label_N]
                                   Position 0: path to .mat file
                                   Position 3: sampling frequency
                                   Position 4+: label values
            ecg_path (str, optional): Base ECG path (deprecated, kept for compatibility)
            target_fs (int): Target sampling frequency for resampling
            transform (callable, optional): Optional transform to be applied
        """
        self.labels_df = labels_df.reset_index(drop=True)
        self.ecg_path = ecg_path
        self.target_fs = target_fs
        self.transform = transform
        
        # Lead configuration for reordering if needed
        self.input_leads = ['I', 'II', 'III', 'aVR', 'aVF', 'aVL', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        self.new_leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        self.lead_indices = [self.input_leads.index(lead) for lead in self.new_leads]

    # =========================================================
    # NORMALIZATION
    # =========================================================
    def z_score_normalization(self, signal):
        """Global z-score normalization across all leads"""
        return (signal - np.mean(signal)) / (np.std(signal) + 1e-8)

    # =========================================================
    # RESAMPLING
    # =========================================================
    def resample_unequal(self, ts, fs_in, fs_out):
        """
        Resample signal from fs_in to fs_out (fixed output length)
        
        Args:
            ts: input signal of shape (n_leads, n_samples)
            fs_in: input sampling frequency
            fs_out: output sampling frequency (or fixed output length)
        
        Returns:
            resampled signal of shape (n_leads, fs_out)
        """
        if fs_in == 0 or len(ts) == 0:
            return ts

        duration = ts.shape[1] / fs_in
        fs_in, fs_out = int(fs_in), int(fs_out)

        if fs_out == fs_in:
            return ts
        
        if 2 * fs_out == fs_in:
            return ts[:, ::2]

        # Create output array with fixed size
        resampled_ts = np.zeros((ts.shape[0], fs_out))
        
        # Create interpolation points
        x_old = np.linspace(0, duration, num=ts.shape[1], endpoint=True)
        x_new = np.linspace(0, duration, num=fs_out, endpoint=True)

        # Interpolate each lead
        for i in range(ts.shape[0]):
            f = interp1d(x_old, ts[i, :], kind='linear')
            resampled_ts[i, :] = f(x_new)

        return resampled_ts

    # =========================================================
    # BANDPASS FILTER (placeholder - implement as needed)
    # =========================================================
    def filter_bandpass(self, signal, fs):
        """
        Optional bandpass filter for ECG (0.5-100 Hz typical range)
        If not needed, this can be removed or left as a no-op
        
        To implement:
        from scipy.signal import butter, filtfilt
        
        def filter_bandpass(self, signal, fs, lowcut=0.5, highcut=100):
            nyquist = fs / 2
            low = lowcut / nyquist
            high = highcut / nyquist
            
            b, a = butter(4, [low, high], btype='band')
            
            filtered_signal = np.zeros_like(signal)
            for i in range(signal.shape[0]):
                filtered_signal[i, :] = filtfilt(b, a, signal[i, :])
            
            return filtered_signal
        """
        # Placeholder: return signal as-is
        return signal

    # =========================================================
    # DATASET LENGTH
    # =========================================================
    def __len__(self):
        return len(self.labels_df)

    # =========================================================
    # MAIN DATA LOADING
    # =========================================================
    def __getitem__(self, idx):
        """
        Load and process a single sample
        
        DataFrame structure expected:
        Position 0: path (str) - path to .mat file
        Position 1: column 1 (unused)
        Position 2: column 2 (unused)
        Position 3: fs (int) - sampling frequency
        Position 4+: label values (float) - label vector
        
        Returns:
            signal: torch.Tensor of shape (12, target_fs)
            labels: torch.Tensor of label vector
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.labels_df.iloc[idx]

        # -------------------------------------------------------
        # Extract metadata from row using positional indexing (iloc)
        # -------------------------------------------------------
        file_path = str(row.iloc[0])  # path (column 0)
        fs_in = int(row.iloc[3])      # fs (column 3)
        labels = row.iloc[4:].values.astype(np.float32)  # all columns from index 4 onwards

        # -------------------------------------------------------
        # Load .mat file
        # -------------------------------------------------------
        try:
            mat_data = scipy.io.loadmat(file_path)
            signal_data = mat_data['val']  # Assume ECG stored in 'val' key
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            # Fallback: return zeros if file cannot be read
            signal_data = np.zeros((12, int(fs_in * 10)))

        # -------------------------------------------------------
        # Ensure shape is (leads, time)
        # -------------------------------------------------------
        if signal_data.shape[0] != 12 and signal_data.shape[1] == 12:
            signal_data = signal_data.T

        # -------------------------------------------------------
        # Handle NaNs and extreme values
        # -------------------------------------------------------
        signal_data = np.nan_to_num(signal_data, nan=0.0, posinf=0.0, neginf=0.0)
        max_safe_value = 1e6
        signal_data = np.clip(signal_data, -max_safe_value, max_safe_value)

        # -------------------------------------------------------
        # Ensure float32 type
        # -------------------------------------------------------
        signal_data = signal_data.astype(np.float32)

        # -------------------------------------------------------
        # Reorder leads if necessary
        # -------------------------------------------------------
        if len(self.lead_indices) == 12:
            signal_data = signal_data[self.lead_indices, :]

        # -------------------------------------------------------
        # Apply bandpass filter (optional)
        # -------------------------------------------------------
        signal_data = self.filter_bandpass(signal_data, fs_in)

        # -------------------------------------------------------
        # Resample to target_fs
        # -------------------------------------------------------
        signal_data = self.resample_unequal(signal_data, fs_in, self.target_fs)

        # -------------------------------------------------------
        # Final safety check for NaNs/Infs
        # -------------------------------------------------------
        signal_data = np.nan_to_num(signal_data, nan=0.0, posinf=0.0, neginf=0.0)

        # -------------------------------------------------------
        # Apply z-score normalization
        # -------------------------------------------------------
        signal_data = self.z_score_normalization(signal_data)

        # -------------------------------------------------------
        # Final validation
        # -------------------------------------------------------
        if not np.isfinite(signal_data).all():
            signal_data = np.zeros_like(signal_data)

        # -------------------------------------------------------
        # Convert to torch tensors
        # -------------------------------------------------------
        signal_tensor = torch.tensor(signal_data, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.float32)

        # -------------------------------------------------------
        # Optional transform
        # -------------------------------------------------------
        if self.transform:
            signal_tensor = self.transform(signal_tensor)

        return signal_tensor, labels_tensor

class LVEF_12lead_reg_Dataset(Dataset):
    def __init__(self, ecg_path, labels_df, transform=None):
        """
        Args:
            labels_df (DataFrame): DataFrame containing the annotations.
            data_dir (str): Directory path containing the numpy data files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.labels_df = labels_df
        self.transform = transform
        self.ecg_path = ecg_path
        self.input_leads = ['I', 'II', 'III', 'aVR', 'aVF', 'aVL', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        self.new_leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'] # leads of MIMIC-ECG are different with leads of HEEDB
        self.lead_indices = [self.input_leads.index(lead) for lead in self.new_leads]

    def __len__(self):
        return len(self.labels_df)

    def z_score_normalization(self,signal):
        return (signal - np.mean(signal)) / (np.std(signal) +1e-8) 

    def check_nan_in_array(self, arr):
        contains_nan = np.isnan(arr).any()
        return contains_nan
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        hash_file_name = str(self.labels_df.iloc[idx, 1])
        labels = self.labels_df.iloc[idx, -2]
        labels = torch.tensor([labels], dtype=torch.float32)  # Wrap the label in a list to create an extra dimension
        data = [wfdb.rdsamp(self.ecg_path + hash_file_name)]
        data = np.array([signal for signal, meta in data])
        data = np.nan_to_num(data, nan=0)
        data = data.squeeze(0)
        data = np.transpose(data, (1, 0))
        data = data[self.lead_indices, :]
        data = filter_bandpass(data, 500) 
        signal = self.z_score_normalization(data)
        signal = torch.FloatTensor(signal)

        return signal, labels     

class LVEF_1lead_cls_Dataset(Dataset):
    def __init__(self, ecg_path, labels_df, transform=None):
        """
        Args:
            labels_df (DataFrame): DataFrame containing the annotations.
            data_dir (str): Directory path containing the numpy data files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.labels_df = labels_df
        self.transform = transform
        self.ecg_path = ecg_path
        self.input_leads = ['I', 'II', 'III', 'aVR', 'aVF', 'aVL', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        self.new_leads = ['I']
        self.lead_indices = [self.input_leads.index(lead) for lead in self.new_leads]

    def __len__(self):
        return len(self.labels_df)

    def z_score_normalization(self,signal):
        return (signal - np.mean(signal)) / (np.std(signal) +1e-8) 

    def check_nan_in_array(self, arr):
        contains_nan = np.isnan(arr).any()
        return contains_nan
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        result = 0
        hash_file_name = str(self.labels_df.iloc[idx, 1])
        labels = self.labels_df.iloc[idx, -1]
        labels = labels.astype(np.float32)
        data = [wfdb.rdsamp(self.ecg_path+hash_file_name)]
        data = np.array([signal for signal, meta in data])
        data = np.nan_to_num(data, nan=0)
        result = self.check_nan_in_array(data)
        if result != 0:
            print(hash_file_name)
        data = data.squeeze(0) 
        data = np.transpose(data,  (1, 0))
        data = data[self.lead_indices, :]
        data = filter_bandpass(data, 500) 
        signal = self.z_score_normalization(data)
        signal = torch.FloatTensor(signal)

        # Convert to torch tensors
        labels = torch.tensor(labels, dtype=torch.float)
        if labels.dim() == 0:  
            labels = labels.unsqueeze(0)
        elif labels.dim() == 1:  
            labels = labels.unsqueeze(1)
        return signal, labels  
    
class LVEF_1lead_reg_Dataset(Dataset):

    def __init__(self, ecg_path, labels_df, transform=None):
        """
        Args:
            labels_df (DataFrame): DataFrame containing the annotations.
            data_dir (str): Directory path containing the numpy data files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.labels_df = labels_df
        self.transform = transform
        self.ecg_path = ecg_path
        self.input_leads = ['I', 'II', 'III', 'aVR', 'aVF', 'aVL', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        self.new_leads = ['I']
        self.lead_indices = [self.input_leads.index(lead) for lead in self.new_leads]

    def __len__(self):
        return len(self.labels_df)

    def z_score_normalization(self,signal):
        return (signal - np.mean(signal)) / (np.std(signal) +1e-8) 

    def check_nan_in_array(self, arr):
        contains_nan = np.isnan(arr).any()
        return contains_nan
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        hash_file_name = str(self.labels_df.iloc[idx, 1])
        labels = self.labels_df.iloc[idx, -2]
        labels = torch.tensor([labels], dtype=torch.float32)  # Wrap the label in a list to create an extra dimension
        data = [wfdb.rdsamp(self.ecg_path + hash_file_name)]
        data = np.array([signal for signal, meta in data])
        data = np.nan_to_num(data, nan=0)
        data = data.squeeze(0)
        data = np.transpose(data, (1, 0))
        data = data[self.lead_indices, :]
        data = filter_bandpass(data, 500) 
        signal = self.z_score_normalization(data)
        signal = torch.FloatTensor(signal)

        return signal, labels     
'''