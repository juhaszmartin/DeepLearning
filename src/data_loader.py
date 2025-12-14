"""Data loading and preprocessing functions for bull flag detection."""

import os
import json
from datetime import datetime
import pandas as pd
from dateutil import parser as date_parser
import uuid


def to_ms(val):
    """Robustly converts timestamps to Unix milliseconds (int64)."""
    if val is None or val == "":
        return None
    if isinstance(val, (int, float)):
        return int(val)
    if isinstance(val, str):
        if val.isdigit():
            return int(val)
        try:
            dt = date_parser.parse(val)
            return int(dt.timestamp() * 1000)
        except Exception:
            return None
    return None


def load_all_labels(root_path):
    """
    Load all label JSON files from student folders.
    
    Args:
        root_path: Root directory containing student folders
        
    Returns:
        DataFrame with columns: annotator, json_file, csv_file, start_raw, start_ms, end_ms, label
    """
    all_segments = []
    
    # 1. Identify valid folders (skipping consensus/sample)
    folders = [f for f in os.listdir(root_path) 
               if os.path.isdir(os.path.join(root_path, f)) 
               and f.lower() not in ("consensus", "sample")]
    
    print(f"Processing {len(folders)} student folders for 'timeserieslabels'...\n")

    for folder in folders:
        folder_path = os.path.join(root_path, folder)
        json_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.json')]
        
        for jf in json_files:
            file_path = os.path.join(folder_path, jf)
            
            try:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    data = json.load(f)
                
                # Standardize data to a list of tasks
                if isinstance(data, dict):
                    data = [data]
                
                for task in data:
                    # --- FORMAT A: Standard Label Studio (annotations -> result -> value) ---
                    if 'annotations' in task:
                        # Extract CSV name
                        csv_name = task.get('data', {}).get('csv') or \
                                   task.get('file_upload') or \
                                   task.get('csv') or \
                                   "unknown_file.csv"
                        csv_name = os.path.basename(csv_name)

                        for ann in task.get('annotations', []):
                            for res in ann.get('result', []):
                                val = res.get('value', {})
                                start_raw = val.get('start')
                                end_raw = val.get('end')
                                
                                # STRICTLY LOOK FOR 'timeserieslabels'
                                # Label Studio returns a list, e.g., ["BearFlag"]
                                label_list = val.get('timeserieslabels')
                                
                                if label_list and isinstance(label_list, list) and len(label_list) > 0:
                                    label_tag = label_list[0]
                                else:
                                    # If key is missing, mark as Unknown so we can debug
                                    label_tag = "Unknown" 

                                if start_raw is not None and end_raw is not None:
                                    all_segments.append({
                                        'annotator': folder,
                                        'json_file': jf,
                                        'csv_file': csv_name,
                                        'start_raw': start_raw,
                                        'start_ms': to_ms(start_raw),
                                        'end_ms': to_ms(end_raw),
                                        'label': label_tag
                                    })

                    # --- FORMAT B: Simple Structure (used by TYEGJ8, VWXUD6) ---
                    elif 'label' in task and isinstance(task['label'], list):
                        csv_name = os.path.basename(task.get('csv', 'unknown_file.csv'))
                        
                        for lbl in task['label']:
                            start_raw = lbl.get('start')
                            end_raw = lbl.get('end')
                            
                            # STRICTLY LOOK FOR 'timeserieslabels' HERE TOO
                            label_list = lbl.get('timeserieslabels')
                            
                            if label_list and isinstance(label_list, list) and len(label_list) > 0:
                                label_tag = label_list[0]
                            else:
                                # Fallback: sometimes these custom formats store label directly as string?
                                # If you see "Unknown" for these folders, we can inspect 'lbl' keys.
                                label_tag = "Unknown"

                            if start_raw is not None:
                                all_segments.append({
                                    'annotator': folder,
                                    'json_file': jf,
                                    'csv_file': csv_name,
                                    'start_raw': start_raw,
                                    'start_ms': to_ms(start_raw),
                                    'end_ms': to_ms(end_raw),
                                    'label': label_tag
                                })

            except Exception as e:
                print(f"Error reading {folder}/{jf}: {e}")

    df = pd.DataFrame(all_segments)
    return df


def find_csv_in_folder(target_folder, filename_from_json):
    """
    Searches for a CSV within a specific student folder.
    Handles 'Smart Matching' to ignore Label Studio UUID prefixes.
    """
    # 1. Try exact match first
    exact_path = os.path.join(target_folder, filename_from_json)
    if os.path.exists(exact_path):
        return exact_path

    # 2. recursive search in student folder (in case of subdirs like 'upload/')
    # AND fuzzy match (ignoring UUID prefix)
    clean_target_name = filename_from_json
    # If the json name has a UUID prefix like "abc12345-file.csv", try to strip it
    if '-' in clean_target_name and clean_target_name[8] == '-': # Simple heuristic for UUID
        clean_target_name = clean_target_name.split('-', 1)[1]

    for root, dirs, files in os.walk(target_folder):
        for f in files:
            # Case A: Exact filename match in subdirectory
            if f == filename_from_json:
                return os.path.join(root, f)
            
            # Case B: The file on disk is "data.csv" but JSON asks for "12345-data.csv"
            if filename_from_json.endswith(f) and len(f) > 4:
                return os.path.join(root, f)

            # Case C: The file on disk has a prefix, or JSON has a prefix (Flexible endswith)
            # This handles: JSON="uuid-data.csv" vs Disk="data.csv"
            if f.endswith(clean_target_name) or clean_target_name.endswith(f):
                 return os.path.join(root, f)
                 
    return None


def build_timeseries_df_local(metadata_df, root_dir):
    """
    Build timeseries DataFrame from metadata and CSV files.
    
    Args:
        metadata_df: DataFrame from load_all_labels()
        root_dir: Root directory containing student folders
        
    Returns:
        DataFrame with timeseries data and segment labels
    """
    all_segments_data = []
    
    # Group by Annotator (Folder) so we look for CSVs in the right place
    grouped = metadata_df.groupby('annotator')
    
    print(f"Processing {len(grouped)} folders...\n")

    for annotator, group in grouped:
        student_folder = os.path.join(root_dir, annotator)
        
        # Get unique CSVs this specific student used
        student_csvs = group['csv_file'].unique()
        
        print(f"{annotator}")
        
        for csv_name in student_csvs:
            # Locate CSV *specifically* in this student's folder
            csv_path = find_csv_in_folder(student_folder, csv_name)
            
            if not csv_path:
                count = len(group[group['csv_file'] == csv_name])
                print(f"   Missing CSV: '{csv_name}' (needed for {count} labels)")
                continue

            try:
                # Load CSV
                df_raw = pd.read_csv(csv_path, sep=None, engine='python')
                
                # Standardize Time Column
                time_col = [c for c in df_raw.columns if 'time' in c.lower()]
                time_col = time_col[0] if time_col else df_raw.columns[0]
                
                # Robust Time Conversion (to match JSON ms)
                try:
                    df_raw['_ts_ms'] = pd.to_numeric(df_raw[time_col])
                    if df_raw['_ts_ms'].mean() < 2e10: # If seconds, convert to ms
                        df_raw['_ts_ms'] = df_raw['_ts_ms'] * 1000
                except:
                    df_raw['_ts_ms'] = pd.to_datetime(df_raw[time_col], utc=True).astype('int64') // 10**6

                # Extract Segments for this specific CSV
                # Filter the group to only rows using this specific CSV
                csv_labels = group[group['csv_file'] == csv_name]
                
                extracted_count = 0
                for _, row in csv_labels.iterrows():
                    start = row['start_ms']
                    end = row['end_ms']
                    
                    # Slice
                    mask = (df_raw['_ts_ms'] >= start) & (df_raw['_ts_ms'] <= end)
                    segment_slice = df_raw[mask].copy()
                    
                    if not segment_slice.empty:
                        segment_slice['segment_id'] = str(uuid.uuid4())
                        segment_slice['label'] = row['label']
                        segment_slice['annotator'] = annotator
                        segment_slice['original_csv'] = csv_name # useful for debug
                        
                        # Normalize columns
                        segment_slice.rename(columns={time_col: 'timestamp'}, inplace=True)
                        segment_slice.columns = [c.lower() for c in segment_slice.columns]
                        
                        all_segments_data.append(segment_slice)
                        extracted_count += 1
                
                print(f"   Loaded '{csv_name}' -> {extracted_count} segments")

            except Exception as e:
                print(f"   Error reading '{csv_name}': {e}")
        
        # visual separator between students
        print("-" * 40)

    # Compile Final DataFrame
    if all_segments_data:
        final_df = pd.concat(all_segments_data, ignore_index=True)
        print(f"\nDONE! Extracted {len(final_df)} rows across {final_df['segment_id'].nunique()} unique segments.")
        return final_df
    else:
        print("\nNo data extracted.")
        return pd.DataFrame()


def remove_outliers(df_timeseries, max_len=100):
    """
    Remove outlier segments that exceed max_len datapoints.
    
    Args:
        df_timeseries: DataFrame with timeseries data
        max_len: Maximum allowed length for a segment
        
    Returns:
        Cleaned DataFrame
    """
    # Calculate length of every segment
    segment_stats = df_timeseries.groupby('segment_id').agg(
        length=('close', 'count'),
        annotator=('annotator', 'first'),
        csv_file=('original_csv', 'first'),
        label=('label', 'first')
    ).reset_index()

    outliers = segment_stats[segment_stats['length'] > max_len].sort_values('length', ascending=False)

    if not outliers.empty:
        print(f"FOUND {len(outliers)} OUTLIER SEGMENTS (Length > {max_len}):")
        print("-" * 80)
        print(f"{'Length':<10} {'Annotator':<15} {'Label':<20} {'CSV File'}")
        print("-" * 80)
        
        for _, row in outliers.iterrows():
            print(f"{row['length']:<10} {row['annotator']:<15} {row['label']:<20} {row['csv_file']}")
        
        print("-" * 80)

        # Filter them out
        outlier_ids = outliers['segment_id'].values
        df_clean = df_timeseries[~df_timeseries['segment_id'].isin(outlier_ids)].copy()
        
        print(f"\nCleaned Dataset: {len(df_clean)} rows (was {len(df_timeseries)})")
        
    else:
        print("No outliers found. Dataset is clean.")
        df_clean = df_timeseries.copy()
    
    return df_clean

