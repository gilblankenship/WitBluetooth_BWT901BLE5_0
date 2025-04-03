#!/usr/bin/env python
"""
Command-line interface for IMU stair movement analysis.
This script analyzes IMU data from a person walking on stairs to calculate step counts,
right-to-left leg differences, and determine stair direction.
"""

import os
import sys
import argparse
from stair_analysis import main as run_analysis

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Analyze IMU data from stair movement to detect steps and leg differences")
    
    parser.add_argument('input_file', 
                        help='Path to the IMU data file (tab-separated format)')
    
    parser.add_argument('-o', '--output', default='output',
                        help='Output directory for results (default: ./output)')
    
    parser.add_argument('--window-size', type=int, default=5,
                        help='Window size for smoothing acceleration data (default: 5)')
    
    parser.add_argument('--threshold-factor', type=float, default=0.5,
                        help='Threshold factor for peak detection (default: 0.5)')
    
    parser.add_argument('--min-peak-distance', type=int, default=5,
                        help='Minimum samples between peaks (default: 5)')
    
    parser.add_argument('--max-time-gap', type=int, default=1000,
                        help='Maximum time gap (ms) for matching steps between devices (default: 1000)')
    
    parser.add_argument('--print-report', action='store_true',
                        help='Print the full report to stdout')
    
    return parser.parse_args()

def main():
    """Main entry point for the CLI"""
    args = parse_args()
    
    # Check if input file exists
    if not os.path.isfile(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found", file=sys.stderr)
        return 1
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    try:
        # Run the analysis
        print(f"Analyzing IMU data from: {args.input_file}")
        results = run_analysis(args.input_file)
        
        # Print summary
        device_names = list(results['device_data'].keys())
        step_counts = {name: len(results['peaks_data'][name]) for name in device_names}
        
        print("\n=== Analysis Summary ===")
        print(f"Total data points: {len(results['df'])}")
        print(f"Devices detected: {', '.join(device_names)}")
        print(f"Total steps detected: {sum(step_counts.values())}")
        
        for name, count in step_counts.items():
            print(f"  - {name}: {count} steps")
        
        print(f"Stair direction: {results['stair_direction']}")
        
        if results['matched_pairs']:
            match_percent = (len(results['matched_pairs']) / min(list(step_counts.values()))) * 100
            print(f"Step matching: {len(results['matched_pairs'])} pairs ({match_percent:.1f}%)")
        
        if results['leg_diff_analysis']:
            print(f"Leg asymmetry: {results['leg_diff_analysis']['acceleration_diff_pct']:.1f}% " 
                  f"acceleration, {results['leg_diff_analysis']['angle_x_diff_pct']:.1f}% angle change")
        
        print(f"\nResults saved to: {args.output}/")
        print(f"Full report: {os.path.join(args.output, 'report.md')}")
        
        # Print full report if requested
        if args.print_report:
            print("\n" + "="*80)
            print("FULL ANALYSIS REPORT")
            print("="*80 + "\n")
            print(results['report'])
        
        return 0
    
    except Exception as e:
        print(f"Error during analysis: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())