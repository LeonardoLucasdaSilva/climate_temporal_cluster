#!/usr/bin/env python
"""
Quick reference and launcher for climate_cluster experiments.

This script provides easy access to all available experiments and utilities.
"""

import subprocess
import sys
from pathlib import Path


def run_experiment(name):
    """Run a named experiment."""
    experiments = {
        "1": ("rs_a801_precipitation_clusters", "RS A801 Precipitation Clustering Analysis"),
    }

    if name not in experiments:
        print("Invalid experiment number")
        return False

    script_name, description = experiments[name]
    script_path = Path(__file__).parent / "experiments" / f"{script_name}.py"

    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"{'='*80}\n")

    try:
        subprocess.run([sys.executable, str(script_path)], check=True)
        return True
    except subprocess.CalledProcessError:
        print(f"\n❌ Experiment failed")
        return False


def run_utility(name):
    """Run a named utility."""
    utilities = {
        "1": ("utils", "Get highest precipitation days (RS A801)"),
    }

    if name not in utilities:
        print("Invalid utility number")
        return False

    script_name, description = utilities[name]
    script_path = Path(__file__).parent / f"{script_name}.py"

    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"{'='*80}\n")

    try:
        subprocess.run([sys.executable, str(script_path)], check=True)
        return True
    except subprocess.CalledProcessError:
        print(f"\n❌ Utility failed")
        return False


def main():
    """Main menu."""
    while True:
        print("\n" + "="*80)
        print("CLIMATE CLUSTER - EXPERIMENTS & UTILITIES")
        print("="*80)
        print("\n[EXPERIMENTS]")
        print("  1. RS A801 Precipitation Clustering Analysis")
        print("\n[UTILITIES]")
        print("  2. Get Highest Precipitation Days")
        print("\n[OTHER]")
        print("  3. View Results")
        print("  4. View Documentation")
        print("  5. Exit")
        print("\n" + "-"*80)

        choice = input("Select option (1-5): ").strip()

        if choice == "1":
            run_experiment("1")
        elif choice == "2":
            run_utility("1")
        elif choice == "3":
            print("\nResults saved to: outputs/rs_a801_precip_analysis/")
            print("  - precipitation_cluster_results.csv")
            print("  - analysis_report.txt")
            results_dir = Path(__file__).parent / "outputs" / "rs_a801_precip_analysis"
            if results_dir.exists():
                for file in results_dir.glob("*"):
                    print(f"\n  📄 {file.name}")
            else:
                print("\n  ⚠️  No results yet. Run experiment first!")
        elif choice == "4":
            print("\nDocumentation Files:")
            print("  - GETTING_STARTED.md     (Quick reference)")
            print("  - experiments/RS_A801_RESULTS.md (Detailed findings)")
            print("  - WINDOW_FEATURES.md     (Feature engineering)")
            print("  - README.md              (Project overview)")
        elif choice == "5":
            print("\nGoodbye!")
            break
        else:
            print("Invalid choice. Try again.")


if __name__ == "__main__":
    main()

