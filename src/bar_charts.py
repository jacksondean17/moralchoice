#!/usr/bin/env python3
"""
Script: analyze_decision_confidence.py

Description:
    This script loads CSV files from a specified folder, each representing responses
    from a single AI model to moral scenarios. For each combination of model_id and scenario_id,
    it calculates a signed confidence value based on the proportion of "action2" decisions.
    If more than 50% of the responses for a (scenario, model) pair are "action2", the confidence
    is positive (indicating preference for action2); if less than 50% then the confidence is negative
    (indicating preference for action1). The script then creates a single combined figure with one
    subplot per scenario. Each subplot shows a bar chart with model_ids on the X-axis and signed confidence
    values on the Y-axis. The combined figure is saved as a PNG file, and summary statistics are printed.

Usage:
    python analyze_decision_confidence.py --data_folder path/to/csv_folder
"""

import os
import glob
import argparse
import math
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def load_data(folder_path):
    """
    Loads all CSV files from the specified folder and returns a combined DataFrame.

    Args:
        folder_path (str): Path to the folder containing CSV files.

    Returns:
        pd.DataFrame: Combined DataFrame with data from all CSV files.
    """
    csv_pattern = os.path.join(folder_path, "*.csv")
    csv_files = glob.glob(csv_pattern)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in folder: {folder_path}")
    
    dataframes = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            dataframes.append(df)
        except Exception as e:
            print(f"Error reading file {file}: {e}")
    
    if not dataframes:
        raise ValueError("No data loaded from CSV files.")
    
    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df

def compute_confidence(df):
    """
    Computes the signed confidence for each (scenario_id, model_id) group.
    For each group:
        - Calculate the percentage of "action2" decisions.
        - If percentage > 50, confidence is positive (preference for action2).
        - If percentage < 50, confidence is negative (preference for action1).
        - If exactly 50%, confidence is 0.

    Args:
        df (pd.DataFrame): DataFrame containing the raw responses.

    Returns:
        pd.DataFrame: DataFrame with columns: scenario_id, model_id, confidence, question_text.
    """
    records = []
    grouped = df.groupby(["scenario_id", "model_id"])
    for (scenario, model), group in grouped:
        total = len(group)
        count_action2 = (group["decision"].str.lower() == "action2").sum()
        pct_action2 = (count_action2 / total) * 100

        # Determine signed confidence based on the dominant action:
        # If action2 percentage > 50, use that percentage (positive);
        # if action2 percentage < 50, then action1 was dominant → use -(100 - pct_action2)
        if pct_action2 > 50:
            confidence = pct_action2
        elif pct_action2 < 50:
            confidence = -(100 - pct_action2)
        else:
            confidence = 0

        sample_question_text = group["question_text"].iloc[0] if "question_text" in group.columns else "No question text available"
        records.append({
            "scenario_id": scenario,
            "model_id": model,
            "confidence": confidence,
            "question_text": sample_question_text
        })
    return pd.DataFrame(records)

def create_combined_bar_chart(conf_df, output_folder="."):
    """
    Creates a single combined figure with a subplot for each scenario_id.
    Each subplot is a bar chart with:
        - X-axis: model_ids.
        - Y-axis: signed confidence values (-100 to +100).
        - Bars colored blue for positive (action2 preference), red for negative (action1 preference),
          and gray if neutral.
        - A horizontal line at y=0 for reference.
        - A title that includes the scenario_id and a brief excerpt from the question_text.
    
    The combined figure is saved as a single PNG file, and summary statistics for each scenario
    are printed to the console.

    Args:
        conf_df (pd.DataFrame): DataFrame with computed confidence values.
        output_folder (str): Folder where the PNG file will be saved.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    scenarios = conf_df["scenario_id"].unique()
    n_scenarios = len(scenarios)
    
    # Determine grid size (for example, up to 3 columns)
    ncols = min(3, n_scenarios)
    nrows = math.ceil(n_scenarios / ncols)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    
    for idx, scenario in enumerate(scenarios):
        ax = axes[idx // ncols][idx % ncols]
        group = conf_df[conf_df["scenario_id"] == scenario]
        models = group["model_id"].tolist()
        # Format model names to show only the last part after splitting by '/'
        display_models = [m.split('/')[-1] for m in models]
        confidences = group["confidence"].tolist()
        colors = ["blue" if conf > 0 else "red" if conf < 0 else "gray" for conf in confidences]
        
        ax.bar(display_models, confidences, color=colors)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylim(-100, 100)
        
        # Rotate x-axis labels
        plt.sca(ax)
        plt.xticks(rotation=45, ha='right')
        
        # Simplified title: just show scenario ID
        ax.set_title(f"Scenario {scenario}", fontsize=10)
        
        ax.set_xlabel("Model ID")
        ax.set_ylabel("Signed Confidence (%)")
        
        # Add a legend
        legend_elements = [
            Patch(facecolor='blue', label='Preference for action2'),
            Patch(facecolor='red', label='Preference for action1'),
            Patch(facecolor='gray', label='Neutral')
        ]
        ax.legend(handles=legend_elements, fontsize=8)
        
        # Print summary statistics for the scenario
        num_models = group["model_id"].nunique()
        conf_min = group["confidence"].min()
        conf_max = group["confidence"].max()
        num_pref_action2 = (group["confidence"] > 0).sum()
        num_pref_action1 = (group["confidence"] < 0).sum()
        print(f"\nSummary for Scenario {scenario}:")
        print(f"  Number of models responded: {num_models}")
        print(f"  Confidence range: {conf_min:.2f} to {conf_max:.2f}")
        print(f"  Models preferring action2: {num_pref_action2}")
        print(f"  Models preferring action1: {num_pref_action1}")
        print("-" * 50)
    
    # Hide any unused subplots if there are any extra axes
    total_subplots = nrows * ncols
    if total_subplots > n_scenarios:
        for idx in range(n_scenarios, total_subplots):
            fig.delaxes(axes[idx // ncols][idx % ncols])
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_file = os.path.join(output_folder, "combined_scenarios.png")
    try:
        plt.savefig(output_file)
        print(f"\nSaved combined plot as {output_file}")
    except Exception as e:
        print(f"Error saving combined plot: {e}")
    finally:
        plt.close()

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Visualize decision confidence across LLMs for moral scenarios.")
    parser.add_argument("--data_folder", type=str, required=True, help="Path to folder containing CSV files.")
    parser.add_argument("--output_folder", type=str, default="plots", help="Folder to save output PNG file.")
    args = parser.parse_args()
    
    # Load data
    try:
        print(f"Loading CSV files from folder: {args.data_folder}")
        data_df = load_data(args.data_folder)
        print(f"Loaded {len(data_df)} records from CSV files.")
    except Exception as e:
        print(f"Failed to load data: {e}")
        return

    # Compute confidence values
    try:
        conf_df = compute_confidence(data_df)
        if conf_df.empty:
            print("No confidence data computed. Exiting.")
            return
    except Exception as e:
        print(f"Error computing confidence: {e}")
        return

    # Create a combined visualization and print summary statistics
    try:
        create_combined_bar_chart(conf_df, output_folder=args.output_folder)
    except Exception as e:
        print(f"Error during visualization: {e}")

if __name__ == '__main__':
    main()
