#!/usr/bin/env python3
# filepath: /workspace/decg/scripts/preprocess_icentia.py

import os
import json
import random
import wfdb
import argparse
from collections import Counter
from tqdm import tqdm

from clef.utils.utils import load_records

def preprocess_icentia_beats(data_path, output_dir, input_len=1000, train_ratio=0.7, val_ratio=0.15):
    """
    Preprocess Icentia ECG data for beat classification and store metadata for fast loading.
    
    Args:
        data_path: Path to raw Icentia data directory
        output_dir: Directory to save preprocessed data and metadata
        input_len: Length of ECG segment to extract
        train_ratio: Ratio of data for training set
        val_ratio: Ratio of data for validation set (test = 1 - train - val)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    patients = list(load_records(os.path.join(data_path, "RECORDS")))
    
    random.seed(42)
    random.shuffle(patients)
    
    # Split patients into train/val/test
    n_patients = len(patients)
    train_size = int(n_patients * train_ratio)
    val_size = int(n_patients * val_ratio)
    
    train_patients = patients[:train_size]
    val_patients = patients[train_size:train_size + val_size]
    test_patients = patients[train_size + val_size:]
    
    print(f"Train patients: {len(train_patients)}")
    print(f"Val patients: {len(val_patients)}")
    print(f"Test patients: {len(test_patients)}")
    
    # Beat types to look for
    beat_types = ['N', 'S', 'V']
    beat_map = {beat: idx for idx, beat in enumerate(beat_types)}
    
    # Process each split
    for split_name, split_patients in [
        ("train", train_patients),
        ("val", val_patients),
        ("test", test_patients)
    ]:
        print(f"\nProcessing {split_name} split...")
        samples = []
        found_beat_types_per_patient = {patient: set() for patient in split_patients}
        
        for patient in tqdm(split_patients, desc=f"Processing {split_name} patients"):
            patient_record = f'{data_path}/{patient}'[:-1]
            
            segments_file = os.path.join(patient_record, "RECORDS")
            if not os.path.exists(segments_file):
                continue
                
            segments = list(load_records(segments_file))
            
            # Skip this patient if we've already found all beat types
            if len(found_beat_types_per_patient[patient]) == len(beat_types):
                continue
                
            for segment in tqdm(segments, desc=f"Processing {patient}", leave=False):
                segment_record = os.path.join(patient_record, segment)
                
                if not os.path.exists(segment_record + ".atr"):
                    continue
                    
                # Skip segments if we've already found all beat types for this patient
                if len(found_beat_types_per_patient[patient]) == len(beat_types):
                    break
                    
                try:
                    # Only search for beat types we haven't found yet
                    remaining_beat_types = [bt for bt in beat_types if bt not in found_beat_types_per_patient[patient]]
                    
                    ann = wfdb.rdann(segment_record, "atr")
                    header = wfdb.rdheader(segment_record)
                    sig_len = header.sig_len
                    
                    # Find indices only for beat types we haven't found yet
                    for beat_type in remaining_beat_types:
                        indices = [i for i, s in enumerate(ann.symbol) if s == beat_type]
                        
                        if indices:  
                            found_beat_types_per_patient[patient].add(beat_type)
                            
                            random_idx = random.choice(indices)
                            sample = ann.sample[random_idx]
                            
                            start = max(0, sample - input_len // 2)
                            end = min(sig_len, sample + input_len // 2)
                            
                            if end - start == input_len:
                                modified_path = segment_record
                                if modified_path.startswith("../../../../"):
                                    modified_path = "../../" + modified_path[12:]

                                samples.append({
                                    'patient': patient,
                                    'segment': segment,
                                    'sample': int(sample),  # Convert numpy values to native Python types for JSON
                                    'signal_path': modified_path,
                                    'start': int(start),
                                    'end': int(end),
                                    'label': beat_map[beat_type]
                                })
                except Exception as e:
                    print(f"Error processing {segment_record}: {e}")
        
        # Print statistics about the collected samples
        beat_type_counts = Counter([beat_types[s['label']] for s in samples])
        print(f"Collected samples by beat type: {dict(beat_type_counts)}")
        print(f"Total samples in {split_name}: {len(samples)}")
        
        # Save metadata as JSON
        output_file = os.path.join(output_dir, f"icentia_beat_{split_name}.json")
        with open(output_file, 'w') as f:
            json.dump(samples, f)
            
        print(f"Saved {split_name} metadata to {output_file}")

def preprocess_icentia_rhythms(data_path, output_dir, input_len=1000, train_ratio=0.7, val_ratio=0.15):
    """
    Preprocess Icentia ECG data for rhythm classification and store metadata for fast loading.
    
    Args:
        data_path: Path to raw Icentia data directory
        output_dir: Directory to save preprocessed data and metadata
        input_len: Length of ECG segment to extract
        train_ratio: Ratio of data for training set
        val_ratio: Ratio of data for validation set (test = 1 - train - val)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    patients = list(load_records(os.path.join(data_path, "RECORDS")))
    
    random.seed(42)
    random.shuffle(patients)
    
    # Split patients into train/val/test
    n_patients = len(patients)
    train_size = int(n_patients * train_ratio)
    val_size = int(n_patients * val_ratio)
    
    train_patients = patients[:train_size]
    val_patients = patients[train_size:train_size + val_size]
    test_patients = patients[train_size + val_size:]
    
    print(f"Train patients: {len(train_patients)}")
    print(f"Val patients: {len(val_patients)}")
    print(f"Test patients: {len(test_patients)}")
    
    # Rhythm types to look for
    rhythm_types = ['(N', '(AFIB', '(AFL']
    rhythm_map = {rhythm: idx for idx, rhythm in enumerate(rhythm_types)}
    
    # Process each split
    for split_name, split_patients in [
        ("train", train_patients),
        ("val", val_patients),
        ("test", test_patients)
    ]:
        print(f"\nProcessing {split_name} split...")
        samples = []
        found_rhythm_types_per_patient = {patient: set() for patient in split_patients}
        
        for patient in tqdm(split_patients, desc=f"Processing {split_name} patients"):
            patient_record = f'{data_path}/{patient}'[:-1]
            
            segments_file = os.path.join(patient_record, "RECORDS")
            if not os.path.exists(segments_file):
                continue
                
            segments = list(load_records(segments_file))
            
            # Skip this patient if we've already found all rhythm types
            if len(found_rhythm_types_per_patient[patient]) == len(rhythm_types):
                continue
                
            for segment in tqdm(segments, desc=f"Processing {patient}", leave=False):
                segment_record = os.path.join(patient_record, segment)
                
                if not os.path.exists(segment_record + ".atr"):
                    continue
                    
                # Skip segments if we've already found all rhythm types for this patient
                if len(found_rhythm_types_per_patient[patient]) == len(rhythm_types):
                    break
                    
                try:
                    # Only search for rhythm types we haven't found yet
                    remaining_rhythm_types = [rt for rt in rhythm_types if rt not in found_rhythm_types_per_patient[patient]]
                    
                    ann = wfdb.rdann(segment_record, "atr")
                    header = wfdb.rdheader(segment_record)
                    sig_len = header.sig_len
                    
                    # Find indices only for rhythm types we haven't found yet
                    for rhythm_type in remaining_rhythm_types:
                        indices = [i for i, s in enumerate(ann.aux_note) if s == rhythm_type]
                        
                        if indices:  
                            found_rhythm_types_per_patient[patient].add(rhythm_type)
                            
                            random_idx = random.choice(indices)
                            sample = ann.sample[random_idx]
                            
                            start = sample
                            end = min(sig_len, sample + input_len)
                            
                            if end - start == input_len:
                                modified_path = segment_record
                                if modified_path.startswith("../../../../"):
                                    modified_path = "../../" + modified_path[12:]

                                samples.append({
                                    'patient': patient,
                                    'segment': segment,
                                    'sample': int(sample),  # Convert numpy values to native Python types for JSON
                                    'signal_path': modified_path,
                                    'start': int(start),
                                    'end': int(end),
                                    'label': rhythm_map[rhythm_type],
                                    'rhythm_type': rhythm_type
                                })
                except Exception as e:
                    print(f"Error processing {segment_record}: {e}")
        
        # Print statistics about the collected samples
        rhythm_type_counts = Counter([rhythm_types[s['label']] for s in samples])
        print(f"Collected samples by rhythm type: {dict(rhythm_type_counts)}")
        print(f"Total samples in {split_name}: {len(samples)}")
        
        # Save metadata as JSON
        output_file = os.path.join(output_dir, f"icentia_rhythm_{split_name}.json")
        with open(output_file, 'w') as f:
            json.dump(samples, f)
            
        print(f"Saved {split_name} metadata to {output_file}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess Icentia ECG data')
    parser.add_argument('--data_path', type=str, default="data/Icentia11K/physionet.org/data", help='Path to Icentia data directory')
    parser.add_argument('--output_dir', type=str, default="dataset/Icentia11K", help='Output directory for preprocessed data')
    parser.add_argument('--input_len', type=int, default=2500, help='Length of ECG segments to extract')
    parser.add_argument('--task', type=str, choices=['beat', 'rhythm', 'all'], default='all', 
                        help='Which task to preprocess data for')
    
    args = parser.parse_args()

    if args.task in ['beat', 'all']:
        beat_output_dir = os.path.join(args.output_dir, 'beat')
        os.makedirs(beat_output_dir, exist_ok=True)
        preprocess_icentia_beats(args.data_path, beat_output_dir, args.input_len)

    if args.task in ['rhythm', 'all']:
        rhythm_output_dir = os.path.join(args.output_dir, 'rhythm')
        os.makedirs(rhythm_output_dir, exist_ok=True)
        preprocess_icentia_rhythms(args.data_path, rhythm_output_dir, args.input_len)
    
    
    
    