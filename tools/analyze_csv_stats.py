import pandas as pd
import numpy as np

df = pd.read_csv('tools/erp_data_export.csv')

# Focus on active window 0.5s to 2.5s
df = df[(df['Time_s'] >= 0.5) & (df['Time_s'] <= 2.5)]

# Calculate Correlations (Pattern Similarity)
corr_c3 = df['C3_Left_V'].corr(df['C3_Right_V'])
corr_c4 = df['C4_Left_V'].corr(df['C4_Right_V'])

# Calculate Mean Absolute Difference (Discriminability Power)
diff_c3 = np.mean(np.abs(df['C3_Left_V'] - df['C3_Right_V']))
diff_c4 = np.mean(np.abs(df['C4_Left_V'] - df['C4_Right_V']))

# Signal Magnitude (Signal Strength)
mag_c3 = np.mean(np.abs(df[['C3_Left_V', 'C3_Right_V']].values))
mag_c4 = np.mean(np.abs(df[['C4_Left_V', 'C4_Right_V']].values))

print(f"ANALYSIS RESULT:")
print(f"C3 Correlation (Left vs Right): {corr_c3:.4f}")
print(f"C4 Correlation (Left vs Right): {corr_c4:.4f}")
print(f"C3 Mean Diff: {diff_c3:.2e} V")
print(f"C4 Mean Diff: {diff_c4:.2e} V")
print(f"Signal Magnitude: ~{mag_c3:.2e} V")

if corr_c3 > 0.8 or corr_c4 > 0.8:
    print("VERDICT: 🔴 High Correlation. The curves are too similar (Noise Dominates).")
elif diff_c3 < 1e-6:
    print("VERDICT: 🔴 Low Difference. The signals are too weak.")
else:
    print("VERDICT: 🟢 Distinct Signals. There is usable information.")
