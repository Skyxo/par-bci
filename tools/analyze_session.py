import pandas as pd
import sys

def analyze_structure(csv_path):
    print(f"--- ANALYSE STRUCTURELLE : {csv_path} ---")
    try:
        # Load timestamps (22) and markers (23)
        df = pd.read_csv(csv_path, sep='\t', header=None, usecols=[22, 23], names=['Time', 'Marker'])
    except Exception as e:
        print(f"Erreur lecture CSV: {e}")
        return

    # Filter all non-zero markers
    events = df[df['Marker'] != 0].copy()
    
    if events.empty:
        print("Aucun marqueur trouvé !")
        return

    # Detect Run boundaries (Marker 99)
    run_starts = [events.index[0]] # Implicit start
    run_ends = events[events['Marker'] == 99].index.tolist()
    
    # Analyze each segment
    current_idx = 0
    run_count = 1
    
    # We iterate through the file based on markers
    # A "Run" is a set of trials. Logic: 99 indicates END of a run (usually).
    
    print(f"\nTotal Markers Detected: {len(events)}")
    print(f"Run Separators (99.0): {len(run_ends)}")
    
    # Manual segmentation based on 99
    # We split the events dataframe by '99'
    
    last_idx = 0
    for end_idx in run_ends:
        # Segment for this run
        segment = events[(events.index >= last_idx) & (events.index <= end_idx)]
        analyze_segment(segment, run_count)
        last_idx = end_idx + 1 # Start next search after this 99
        run_count += 1
        
    # Check for trailing markers after the last 99
    if last_idx < events.index[-1]:
        segment = events[events.index >= last_idx]
        print("\n--- RÉSIDU / RUN INCOMPLET (Après dernier 99) ---")
        analyze_segment(segment, run_count)

def analyze_segment(segment, run_id):
    start_t = segment['Time'].iloc[0]
    end_t = segment['Time'].iloc[-1]
    duration = end_t - start_t
    
    counts = segment['Marker'].value_counts()
    
    # Filter only class markers for clarity
    classes = {1: 'Left', 2: 'Right', 3: 'Feet', 10: 'Rest'}
    
    print(f"\n>>> RUN {run_id}")
    print(f"    Durée estimée: {duration:.1f} sec ({duration/60:.1f} min)")
    print(f"    Indices lignes: {segment.index[0]} -> {segment.index[-1]} (Gap: {segment.index[-1] - segment.index[0]} lignes)")
    
    class_msg = []
    total_trials = 0
    
    for m, name in classes.items():
        if m in counts:
            c = counts[m]
            class_msg.append(f"{name}: {c}")
            total_trials += c
            
    print(f"    Total Essais: {total_trials}")
    print(f"    Détail: {', '.join(class_msg)}")
    
    if total_trials == 24:
        print("    STATUS: ✅ COMPLET")
    else:
        print(f"    STATUS: ⚠️ ANORMAL (Attendu 24, Reçu {total_trials})")

if __name__ == "__main__":
    analyze_structure(sys.argv[1])
