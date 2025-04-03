import os
import numpy as np
import matplotlib.pyplot as plt
from stair_analysis import main as analyze_imu_data

# Run the full analysis
results = analyze_imu_data("paste.txt")

# Extract key results
stair_direction = results['stair_direction']
device_data = results['device_data']
device_names = list(device_data.keys())
step_counts = {name: len(results['peaks_data'][name]) for name in device_names}
leg_diff = results['leg_diff_analysis']

# Display key findings
print("\n=== Key Findings ===")
print(f"Stair direction: {stair_direction}")
print(f"Step counts: {step_counts}")
print(f"Left-right leg differences:")
print(f"  - Acceleration difference: {leg_diff['acceleration_diff_pct']:.2f}%")
print(f"  - Angle change difference: {leg_diff['angle_x_diff_pct']:.2f}%")

# Show dominant leg information if available
if results['matched_pairs']:
    dominant_leg = device_names[0] if leg_diff['dominant_leg'] == 'device1' else device_names[1]
    print(f"Dominant leg (typically moves first): {dominant_leg}")
    print(f"Dominance percentage: {leg_diff['leg_dominance_pct']:.2f}%")

# Check for asymmetry
asymmetry_threshold = 20  # percent
if (leg_diff['acceleration_diff_pct'] > asymmetry_threshold or 
    leg_diff['angle_x_diff_pct'] > asymmetry_threshold):
    print("\nSignificant leg asymmetry detected!")
    
    if leg_diff['acceleration_diff_pct'] > asymmetry_threshold:
        higher_acc_device = device_names[0] if leg_diff['higher_acc_device'] == 'device1' else device_names[1]
        print(f"  - {higher_acc_device} shows {leg_diff['acceleration_diff_pct']:.2f}% higher acceleration")
    
    if leg_diff['angle_x_diff_pct'] > asymmetry_threshold:
        larger_angle_device = device_names[0] if leg_diff['larger_angle_device'] == 'device1' else device_names[1]
        print(f"  - {larger_angle_device} shows {leg_diff['angle_x_diff_pct']:.2f}% larger angle changes")
    
    print("This may indicate uneven stair climbing technique or potential mobility issues.")

# Summarize step consistency for each leg
print("\nStep consistency:")
for i, device_name in enumerate(device_names):
    consistency = results['consistency_data'][device_name]
    if consistency:
        variability = consistency['variability_coefficient']
        print(f"  - {device_name}: {variability:.2f}% variability")
        if variability > 50:
            print(f"    (High variability may indicate irregular stepping pattern)")

# Display timing of steps
print("\nStep timing:")
for i, device_name in enumerate(device_names):
    consistency = results['consistency_data'][device_name]
    if consistency:
        mean_gap = consistency['mean_time_between_steps_ms']
        print(f"  - {device_name}: {mean_gap:.2f}ms between steps")

# Print path to full report
print("\nComplete analysis report available at: output/report.md")
print("Visualizations available in the output/ directory")