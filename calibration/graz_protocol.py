import time
import random
from pylsl import StreamInfo, StreamOutlet
from colorama import Fore, Style, init

# Initialize colorama for console-based visual cues
init()

# ==========================================
# 1. SETUP LSL MARKER STREAM
# ==========================================
# We send "Markers" to the same timeline as the OpenBCI data.
# When you analyze data later, you match these markers to the EEG timestamps.
info = StreamInfo(name='BCI_Markers', type='Markers', channel_count=1,
                  nominal_srate=0, channel_format='string', source_id='bci_calib_001')
outlet = StreamOutlet(info)

print("Checklist:")
print("1. OpenBCI GUI is running.")
print("2. LSL Stream is ACTIVE in OpenBCI GUI.")
print("3. User is seated and relaxed.")
input("Press Enter to start the calibration session...")

# ==========================================
# 2. EXPERIMENT PARAMETERS
# ==========================================
classes = ['Left', 'Right']
n_trials_per_class = 20  # Total 40 trials (approx 5-6 mins)
trials = classes * n_trials_per_class
random.shuffle(trials)

# ==========================================
# 3. THE GRAZ PROTOCOL LOOP
# ==========================================
print("\n--- STARTING SESSION IN 5 SECONDS ---")
time.sleep(5)

for i, trial_class in enumerate(trials):
    print(f"\nTrial {i+1}/{len(trials)}")
    
    # --- PHASE 1: FIXATION (t=0 to t=2s) ---
    # User focuses eyes, stops blinking.
    print(Style.RESET_ALL + "  +  (Fixate)")
    outlet.push_sample(['Fixation_Start'])
    time.sleep(2.0)
    
    # --- PHASE 2: CUE PRESENTATION (t=2s to t=3s) ---
    # Show direction.
    if trial_class == 'Left':
        visual = Fore.RED + "<--- LEFT"
        marker = 'Cue_Left'
    else:
        visual = Fore.GREEN + "RIGHT --->"
        marker = 'Cue_Right'
        
    print(visual)
    outlet.push_sample([marker])
    time.sleep(1.0)
    
    # --- PHASE 3: MOTOR IMAGERY (t=3s to t=6s) ---
    # User imagines the movement CONTINUOUSLY here.
    print(visual + Style.BRIGHT + "  (IMAGINE!)") 
    # (No new marker needed, we process data relative to 'Cue' marker)
    time.sleep(3.0)
    
    # --- PHASE 4: BREAK/REST (t=6s to t=8s+) ---
    # Random jitter to prevent rhythm anticipation
    print(Style.RESET_ALL + "     (Rest / Blink)")
    outlet.push_sample(['Rest_Start'])
    time.sleep(2.0 + random.random()) 

print(Style.RESET_ALL + "\n--- SESSION COMPLETE ---")
