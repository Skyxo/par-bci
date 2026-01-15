import pandas as pd
import numpy as np
import glob
import os

def analyze_file(fname):
    print(f"\n🔍 ANALYZING: {os.path.basename(fname)}")
    try:
        try:
            df = pd.read_csv(fname, sep='\t', header=None)
        except:
            df = pd.read_csv(fname, sep=',', header=None)
            
        print(f"   Shape: {df.shape}")
        
        # Check Timestamp (Col 0)
        timestamps = df.iloc[:, 0].values
        fs_est = 1 / np.mean(np.diff(timestamps))
        print(f"   Estimated Fs: {fs_est:.2f} Hz")
        
        # Check Markers (Col 23 or last)
        markers = df.iloc[:, 23].values
        
        # Find changes
        diff_markers = np.diff(markers, prepend=0)
        # Starts
        starts = np.where((diff_markers != 0) & (markers != 0))[0]
        # Ends (next change)
        ends = []
        for s in starts:
            # Find next change index after s
            next_changes = np.where(diff_markers[s+1:] != 0)[0]
            if len(next_changes) > 0:
                ends.append(s + 1 + next_changes[0])
            else:
                ends.append(len(markers)) # To end of file
        
        ends = np.array(ends)
        vals = markers[starts]
        
        print(f"   Found {len(starts)} markers.")
        if len(starts) > 0:
            print(f"   First 5 markers: {vals[:5]}")
            durations = ends - starts
            durations_sec = durations / fs_est
            print(f"   Durations (samples): Min={min(durations)}, Max={max(durations)}, Mean={np.mean(durations):.1f}")
            print(f"   Durations (sec):     Min={min(durations_sec):.3f}s, Max={max(durations_sec):.3f}s")
            
            # Check for very short markers
            shorties = np.sum(durations_sec < 0.5)
            if shorties > 0:
                print(f"   ⚠️ WARNING: {shorties} markers are shorter than 0.5s!")
                
    except Exception as e:
        print(f"   ❌ Error: {e}")

files = glob.glob("EEG_Session_*.csv")
for f in files:
    analyze_file(f)
