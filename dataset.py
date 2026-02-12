import os
import numpy as np
import torch
import mne
import pandas as pd
import glob
import re
from torch.utils.data import Dataset

RAW_FOLDER = "./raw_data"        
PROCESSED_FOLDER = "./processed_data" 

TARGET_SR = 256  
WINDOW_SEC = 4
STRIDE_SEC = 1
PATCH_SIZE = 64  

STANDARD_CHANNELS = [
    'FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
    'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'FZ', 'CZ', 'PZ',
    'FPZ', 'AF3', 'AF4', 'F1', 'F2', 'F5', 'F6', 'FC1', 'FC2', 'FC3', 
    'FC4', 'FC5', 'FC6', 'FT7', 'FT8', 'C1', 'C2', 'C5', 'C6', 'CP1', 
    'CP2', 'CP3', 'CP4', 'CP5', 'CP6', 'TP7', 'TP8', 'P1', 'P2', 'P5', 
    'P6', 'PO3', 'PO4', 'PO7', 'PO8', 'FT9', 'FT10', 'TP9', 'TP10' 
]

TUSZ_BG_LABELS = {'bckg', 'impls', 'artf', 'eyem', 'chew', 'shiv', 'musc', 'elpp', 'elst'}

class SmartAnnotationParser:
    @staticmethod
    def parse(annotation_path, target_filename=None):
        ext = os.path.splitext(annotation_path)[1].lower()
        
        if ext in ['.csv', '.txt'] and 'summary' not in os.path.basename(annotation_path).lower():
            return SmartAnnotationParser._parse_csv(annotation_path)
        
        elif ext == '.tse':
            return SmartAnnotationParser._parse_tse(annotation_path)
            
        elif ext == '.txt' or 'summary' in os.path.basename(annotation_path).lower():
            if target_filename:
                return SmartAnnotationParser._parse_text_summary(annotation_path, target_filename)
        
        return None

    @staticmethod
    def _parse_csv(path):
        intervals = set()
        try:
            df = pd.read_csv(path, comment='#', header=None)
            for _, row in df.iterrows():
                try:
                    start, stop = float(row.iloc[1]), float(row.iloc[2])
                    label = str(row.iloc[3]).lower().strip()
                    
                    if label not in TUSZ_BG_LABELS:
                        intervals.add((start, stop))
                except (ValueError, IndexError):
                    continue
        except Exception:
            pass
        return list(intervals)

    @staticmethod
    def _parse_tse(path):
        intervals = set()
        try:
            with open(path, 'r') as f:
                for line in f:
                    if line.startswith('version'): continue
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        start, stop = float(parts[0]), float(parts[1])
                        label = parts[2].lower()
                        if label not in TUSZ_BG_LABELS:
                            intervals.add((start, stop))
        except Exception: 
            pass
        return list(intervals)

    @staticmethod
    def _parse_text_summary(path, target_file):
        intervals = []
        file_found = False
        
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            is_target = False
            for i, line in enumerate(lines):
                line = line.strip()
                
                if "File Name:" in line:
                    current_file = line.split(": ")[1].strip()
                    if current_file == target_file:
                        is_target = True
                        file_found = True
                    else:
                        is_target = False
                
                if is_target and line.startswith("Seizure") and "Start Time" in line:
                    try:
                        start = int(re.search(r':\s*(\d+)\s*seconds', line).group(1))
                        end_line = lines[i+1].strip()
                        end = int(re.search(r':\s*(\d+)\s*seconds', end_line).group(1))
                        intervals.append((start, end))
                    except: pass
                    
        except Exception:
            return None

        if file_found:
            return intervals
        else:
            return None

class EpilepsyIngestor:
    def __init__(self, save_path):
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

    def process_file(self, edf_path, seizure_intervals):
        try:
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
            
            if raw.info['sfreq'] != TARGET_SR:
                raw.resample(TARGET_SR)
            
            data = raw.get_data() * 1e6 
            standard_data = np.zeros((len(STANDARD_CHANNELS), data.shape[1]), dtype=np.float32)
            
            raw_ch_map = {ch.upper().split('-')[0].replace('EEG ', '').strip(): i for i, ch in enumerate(raw.ch_names)}
            
            matched_count = 0
            for i, std_ch in enumerate(STANDARD_CHANNELS):
                if std_ch in raw_ch_map:
                    standard_data[i] = data[raw_ch_map[std_ch]]
                    matched_count += 1
                elif std_ch == 'T3' and 'T7' in raw_ch_map: standard_data[i] = data[raw_ch_map['T7']]
                elif std_ch == 'T4' and 'T8' in raw_ch_map: standard_data[i] = data[raw_ch_map['T8']]
                elif std_ch == 'T5' and 'P7' in raw_ch_map: standard_data[i] = data[raw_ch_map['P7']]
                elif std_ch == 'T6' and 'P8' in raw_ch_map: standard_data[i] = data[raw_ch_map['P8']]

            if matched_count < 10:
                print(f"Skipping {os.path.basename(edf_path)}: Too few matching channels ({matched_count}/58)")
                return

            n_samples = data.shape[1]
            labels = np.zeros(n_samples, dtype=np.float32)
            for start, end in seizure_intervals:
                s, e = int(start*TARGET_SR), int(end*TARGET_SR)
                labels[max(0, s):min(n_samples, e)] = 1.0

            self._save_windows(standard_data, labels, os.path.basename(edf_path))
            
        except Exception as e:
            print(f"Error reading {os.path.basename(edf_path)}: {e}")

    def _save_windows(self, data, labels, base_name):
        for i in range(len(data)):
            if np.std(data[i]) > 0:
                data[i] = (data[i] - np.mean(data[i])) / np.std(data[i])

        win_size = int(WINDOW_SEC * TARGET_SR)
        stride = int(STRIDE_SEC * TARGET_SR)
        n_samples = data.shape[1]

        if n_samples < win_size:
            pad_width = win_size - n_samples
            data = np.pad(data, ((0, 0), (0, pad_width)), mode='constant')
            labels = np.pad(labels, (0, pad_width), mode='constant', constant_values=0)
            n_samples = data.shape[1]

        n_wins = 1 + (n_samples - win_size) // stride
        
        saved_count = 0
        seizure_frames = 0
        
        for i in range(n_wins):
            start = i * stride
            stop = start + win_size
            x = data[:, start:stop]
            y = labels[start:stop]
            
            path = os.path.join(self.save_path, f"{base_name.replace('.edf','')}_{i}.npz")
            np.savez_compressed(path, x=x, y=y)
            saved_count += 1
            if np.sum(y) > 0: seizure_frames += 1
            
        print(f"Processed {base_name}: {saved_count} windows ({seizure_frames} seizures)")

class SeizureDataset(Dataset):
    def __init__(self, data_dir, patient_ids=None):
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        if patient_ids:
            self.files = [f for f in self.files if any(pid in f for pid in patient_ids)]
        
        self.samples_per_window = int(WINDOW_SEC * TARGET_SR)
        self.num_tokens = self.samples_per_window // PATCH_SIZE
    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        item = np.load(self.files[idx])
        x = torch.tensor(item['x'], dtype=torch.float32)
        y_raw = item['y']
        
        y_reshaped = y_raw[:self.num_tokens * PATCH_SIZE].reshape(self.num_tokens, PATCH_SIZE)
        y_token = np.max(y_reshaped, axis=1) 
        
        return x, torch.tensor(y_token, dtype=torch.float32)

def auto_discover_and_process(root_dir):
    print(f"Scanning {root_dir} for datasets...")
    ingestor = EpilepsyIngestor(PROCESSED_FOLDER)
    
    edf_files = glob.glob(os.path.join(root_dir, "**/*.edf"), recursive=True)
    print(f"Found {len(edf_files)} EDF files. Beginning smart pairing...")
    
    for edf_path in edf_files:
        folder = os.path.dirname(edf_path)
        filename = os.path.basename(edf_path)
        base_name = os.path.splitext(filename)[0]
        
        intervals = []
        source_found = "None"
        
        possible_annotations = [
            os.path.join(folder, base_name + ".csv"),
            os.path.join(folder, base_name + ".tse"),
            os.path.join(folder, base_name + ".seizures"),
            os.path.join(folder, base_name + ".csv_bi")
        ]
        
        for ann_path in possible_annotations:
            if os.path.exists(ann_path):
                found = SmartAnnotationParser.parse(ann_path)
                if found is not None:
                    intervals = found
                    source_found = os.path.basename(ann_path)
                    break
        
        if source_found == "None":
            txt_files = glob.glob(os.path.join(folder, "*.txt"))
            for txt in txt_files:
                found_intervals = SmartAnnotationParser.parse(txt, target_filename=filename)
                
                if found_intervals is not None:
                    intervals = found_intervals
                    source_found = os.path.basename(txt)
                    break 

        if source_found != "None":
            print(f"Processing {filename} (Source: {source_found}) -> {len(intervals)} sz")
            ingestor.process_file(edf_path, intervals)
        else:
            print(f"Processing {filename} (No annotation found - Assuming Background)")
            ingestor.process_file(edf_path, [])

if __name__ == "__main__":
    auto_discover_and_process(RAW_FOLDER)