import os
import mne
import pandas as pd
from collections import Counter

DATASET_DIR = './Dataset/Infants_data'
OUTPUT_DIR = './Dataset/Dataset_info'

def find_edf_files(root_dir):
    edf_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith('.edf'):
                edf_files.append(os.path.join(dirpath, fname))
    return edf_files

def extract_metadata(edf_path):
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
        info = raw.info
        duration = raw.n_times / info['sfreq']
        # Subject ID extraction: find folder starting with 'sub-'
        parts = edf_path.split(os.sep)
        subject_id = next((p for p in parts if p.startswith('sub-')), None)
        metadata = {
            'subject_id': subject_id,
            'file_path': edf_path,
            'duration_sec': duration,
            'sfreq': info['sfreq'],
            'n_channels': info['nchan'],
            'channel_names': info['ch_names'],
            'start_datetime': info['meas_date'].isoformat() if info['meas_date'] else None
        }
        return metadata
    except Exception as e:
        print(f"Error reading {edf_path}: {e}")
        return None

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    edf_files = find_edf_files(DATASET_DIR)
    metadata_list = [extract_metadata(f) for f in edf_files]
    metadata_list = [m for m in metadata_list if m is not None]
    df = pd.DataFrame(metadata_list)

    # Print summary statistics
    print(f"Total subjects: {df['subject_id'].nunique()}")
    print(f"Total recordings: {len(df)}")
    print(f"Average recording duration (sec): {df['duration_sec'].mean():.2f}")
    mode_sfreq = df['sfreq'].mode()
    print(f"Most common sampling rate: {mode_sfreq.iloc[0] if not mode_sfreq.empty else 'N/A'}")
    print("Distribution of number of channels:")
    print(df['n_channels'].value_counts())

    # Save metadata per subject in JSON
    import json
    for subject_id, subject_df in df.groupby('subject_id'):
        sessions = subject_df[['file_path', 'duration_sec', 'sfreq', 'n_channels', 'channel_names', 'start_datetime']].to_dict(orient='records')
        subject_json = {
            'subject_id': subject_id,
            'sessions': sessions
        }
        json_path = os.path.join(OUTPUT_DIR, f'{subject_id}_metadata.json')
        with open(json_path, 'w') as f:
            json.dump(subject_json, f, indent=2)

    # Save all metadata in one CSV file
    df.to_csv(os.path.join(OUTPUT_DIR, 'all_metadata.csv'), index=False)

if __name__ == "__main__":
    main()