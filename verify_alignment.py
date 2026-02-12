import mne
import numpy as np
import matplotlib.pyplot as plt
import os

# --- CONFIG ---
RAW_PATH = "./raw_data/chb01_03.edf"       # Path to a raw file
PROCESSED_PATH = "./processed_data/chb01_03_0.npz" # Path to its first window
TARGET_SR = 256 # Must match your dataset.py

# The index we want to check (0 is usually FP1 in our standard list)
CHECK_INDEX = 0 
CHECK_NAME = "FP1" 

def check_alignment():
    print(f"Checking alignment for channel: {CHECK_NAME} (Index {CHECK_INDEX})")
    
    # 1. Load Processed Data (The Tensor)
    if not os.path.exists(PROCESSED_PATH):
        print(f"Error: Could not find processed file at {PROCESSED_PATH}")
        print("Run dataset.py first!")
        return

    data = np.load(PROCESSED_PATH)
    tensor_segment = data['x'][CHECK_INDEX, :] # Get Row 0
    print(f"Loaded Tensor shape: {data['x'].shape}")

    # 2. Load Raw Data (The Truth)
    raw = mne.io.read_raw_edf(RAW_PATH, preload=True, verbose=False)
    
    # Find the channel in the raw file (handling "EEG FP1-REF" mess)
    raw_pick = None
    for ch in raw.ch_names:
        if CHECK_NAME in ch.upper().replace('EEG', '').replace('REF', '').replace('-', ''):
            raw_pick = ch
            break
            
    if raw_pick is None:
        print(f"Could not find {CHECK_NAME} in raw file channels: {raw.ch_names}")
        return

    print(f"Found corresponding raw channel: {raw_pick}")
    
    # Resample and crop to match the first window (0 to 4s)
    if raw.info['sfreq'] != TARGET_SR:
        raw.resample(TARGET_SR)
    
    raw_data = raw.get_data(picks=[raw_pick])
    raw_segment = raw_data[0, :len(tensor_segment)] # Grab first 4 seconds
    
    # 3. Normalize Raw for comparison (Z-score)
    # The processed data is Z-scored, so we must Z-score raw to compare shape
    raw_segment = (raw_segment - np.mean(raw_segment)) / (np.std(raw_segment) + 1e-6)

    # 4. Plot Comparison
    plt.figure(figsize=(10, 5))
    plt.title(f"Channel Alignment Check: {CHECK_NAME}")
    
    # Plot Raw (Blue line)
    plt.plot(raw_segment, label=f"Raw EDF ({raw_pick})", alpha=0.7, linewidth=2)
    
    # Plot Tensor (Red dashed)
    plt.plot(tensor_segment, label=f"Processed Tensor (Row {CHECK_INDEX})", color='red', linestyle='--')
    
    plt.legend()
    plt.xlabel("Time (Samples)")
    plt.ylabel("Normalized Amplitude")
    plt.grid(True)
    plt.show()
    
    print("Check the plot:")
    print("✅ If lines overlap perfectly: Channels are ALIGNED.")
    print("❌ If lines are different: MAPPING ERROR.")

if __name__ == "__main__":
    check_alignment()