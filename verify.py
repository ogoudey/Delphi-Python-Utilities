import pandas as pd
from pathlib import Path
import sys
# Replace with the path to one of your parquet files
# e.g., "data/chunk-000.parquet" inside your LeRobot dataset directory
parquet_file_path = Path(sys.argv[1])

def verify_parquet(path: Path):
    print(f"--- Verifying Parquet: {path.name} ---")
    
    # 1. Read the parquet file using pandas/pyarrow
    df = pd.read_parquet(path)
    
    # 2. Print basic info
    print(f"Total Rows (Frames): {len(df)}")
    print(f"Columns found: {list(df.columns)}")
    
    # 3. Print a clean, non-crashing head (first 5 rows)
    # We convert array columns to strings/tuples so print() or tabulate won't crash
    df_printable = df.copy()
    for col in df_printable.columns:
        # Check if the column contains arrays/lists
        if df_printable[col].apply(lambda x: isinstance(x, (list, pd.Series, bytes)) or hasattr(x, '__len__')).any():
            # Summarize the array size so it doesn't clutter the screen
            df_printable[col] = df_printable[col].apply(lambda x: f"Array(shape={len(x)})" if hasattr(x, '__len__') else x)
            
    print("\n--- First 5 Rows (Cleaned Summary) ---")
    print(df_printable.head(5).to_string(index=True))
    
    # 4. Verify indexing continuity
    if "episode_index" in df.columns:
        unique_episodes = df["episode_index"].unique()
        print(f"\nUnique Episode Indices in this chunk: {unique_episodes}")
        print(f"Number of episodes: {len(unique_episodes)}")
        
        # Check for any gaps in frame indexing
        for ep in unique_episodes:
            ep_df = df[df["episode_index"] == ep]
            min_frame = ep_df["index"].min() if "index" in ep_df.columns else ep_df.index.min()
            max_frame = ep_df["index"].max() if "index" in ep_df.columns else ep_df.index.max()
            print(f"  -> Episode {ep}: Frames range from {min_frame} to {max_frame} ({len(ep_df)} frames)")

def verify_parquet_with_language(path: Path):
    print(f"--- Verifying Parquet: {path.name} ---")
    
    # Read the parquet file
    df = pd.read_parquet(path)
    
    # 1. Identify any language/instruction columns
    lang_cols = [col for col in df.columns if "language" in col or "instruction" in col]
    print(f"Found language-related columns: {lang_cols}")
    
    # 2. Clean up columns for visual display (summarizing nested arrays)
    df_printable = df.copy()
    for col in df_printable.columns:
        # If the column contains nested lists/arrays (excluding plain strings)
        if df_printable[col].apply(lambda x: isinstance(x, (list, pd.Series, bytes)) or (hasattr(x, '__len__') and not isinstance(x, str))).any():
            df_printable[col] = df_printable[col].apply(lambda x: f"Array(shape={len(x)})" if hasattr(x, '__len__') else x)
            
    print("\n--- Sample DataFrame Rows ---")
    cols_to_show = [c for c in ["index", "episode_index"] if c in df.columns] + lang_cols
    if cols_to_show:
        print(df_printable[cols_to_show].head(5).to_string(index=True))
    else:
        print(df_printable.head(5).to_string(index=True))
        
    # 3. Extract a single representative language_instruction element safely
    print("\n--- Representative Language Instruction ---")
    for col in lang_cols:
        # Safely convert entries to standard strings (handling list-wrapped strings)
        def extract_string(val):
            if isinstance(val, (list, pd.Series, numpy_ndarray := type(pd.Series().values))):
                return str(val[0]) if len(val) > 0 else ""
            return str(val) if pd.notna(val) else ""

        # Apply standard string extraction and filter out empty strings
        string_series = df[col].apply(extract_string)
        non_empty_strings = string_series[string_series != ""]
        
        if not non_empty_strings.empty:
            representative_value = non_empty_strings.iloc[0]
            print(f"Column '{col}' first representative string: '{representative_value}'")
        else:
            print(f"Column '{col}' is empty or contains only null/empty instructions.")

verify_parquet(parquet_file_path)
print("\n\n\nNEXT\n\n\n")
verify_parquet_with_language(parquet_file_path)