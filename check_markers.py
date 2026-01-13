import pandas as pd
import numpy as np

FILE = 'EEG_Session_2026-01-13_15-30.csv'

print(f"Reading {FILE}...")
try:
    df = pd.read_csv(FILE, sep='\t', header=None)
except:
    df = pd.read_csv(FILE, sep=',', header=None)

markers = df.iloc[:, 23].values
unique, counts = np.unique(markers, return_counts=True)

print("\n--- Marker Counts ---")
for u, c in zip(unique, counts):
    print(f"Marker {int(u)}: {c}")

# Check event counts (transitions)
diff_markers = np.diff(markers, prepend=0)
# Events are where marker changes from 0 to something else AND that something else is 1,2,3,10
events_idx = np.where(np.isin(markers, [1, 2, 3, 10]) & (diff_markers != 0))[0]
event_vals = markers[events_idx]

u_ev, c_ev = np.unique(event_vals, return_counts=True)
print("\n--- Event Counts (Transitions from 0) ---")
for u, c in zip(u_ev, c_ev):
    print(f"Event {int(u)}: {c}")
