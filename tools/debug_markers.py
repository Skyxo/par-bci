import pandas as pd
import matplotlib.pyplot as plt
import sys

def plot_markers(csv_path):
    print(f"Analyzing {csv_path}...")
    # Load only the marker column (index 23) and timestamp (index 22)
    # Using python engine for flexibility with potential bad lines
    try:
        df = pd.read_csv(csv_path, sep='\t', header=None, usecols=[22, 23], names=['Time', 'Marker'])
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Filter non-zero markers
    events = df[df['Marker'] != 0]
    
    if events.empty:
        print("No markers found!")
        return

    print("Marker Counts:")
    print(events['Marker'].value_counts().sort_index())

    # Map markers to labels and colors
    marker_map = {1: 'Left', 2: 'Right', 3: 'Feet', 10: 'Rest', 99: 'EndRun'}
    colors = {1: 'red', 2: 'blue', 3: 'green', 10: 'grey', 99: 'black'}

    plt.figure(figsize=(15, 6))
    
    # Plot each event type
    for m in events['Marker'].unique():
        subset = events[events['Marker'] == m]
        label = marker_map.get(m, str(m))
        color = colors.get(m, 'purple')
        plt.scatter(subset['Time'], [m]*len(subset), label=f"{label} ({m})", color=color, s=100, alpha=0.7)

    plt.yticks(list(marker_map.keys()), list(marker_map.values()))
    plt.xlabel("Timestamp (Unix)")
    plt.title(f"Marker Timeline: {csv_path}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_file = "marker_timeline.png"
    plt.savefig(output_file)
    print(f"Timeline plot saved to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        plot_markers(sys.argv[1])
    else:
        print("Usage: python script.py <csv_file>")
