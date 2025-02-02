#!/usr/bin/env python3
"""
disagreement_analysis.py

This script analyzes moral decision-making patterns across different LLMs.
It reads multiple CSV files (each representing responses from one model) from a specified folder,
calculates decision percentages per scenario/model, identifies strong preferences (≥75%),
and then filters for scenarios where at least two models have strong preferences with opposing choices.
The final results are saved to 'moral_disagreements_analysis.csv' and progress is printed to the console.
"""

import os
import glob
import argparse
import pandas as pd


def load_csv_files(folder_path):
    """
    Load and combine all CSV files in the specified folder into a single DataFrame.
    Performs basic error handling for missing files or columns.
    """
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    if not csv_files:
        print(f"[ERROR] No CSV files found in folder: {folder_path}")
        return None, 0

    data_frames = []
    files_processed = 0
    required_columns = {"scenario_id", "model_id", "question_text", "decision"}
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            # Ensure the file contains the required columns
            missing_cols = required_columns - set(df.columns)
            if missing_cols:
                print(f"[WARNING] File '{file}' is missing columns: {missing_cols}. Skipping this file.")
                continue
            data_frames.append(df)
            files_processed += 1
        except Exception as e:
            print(f"[ERROR] Problem reading file '{file}': {e}")

    if not data_frames:
        print("[ERROR] No valid CSV files were processed.")
        return None, 0

    try:
        combined_df = pd.concat(data_frames, ignore_index=True)
    except Exception as e:
        print(f"[ERROR] Failed to combine CSV files: {e}")
        return None, 0

    return combined_df, files_processed


def compute_strong_preferences(df, threshold=75):
    """
    For each (scenario_id, model_id) combination, calculate the percentage of responses for
    each decision and flag strong preferences if one decision was chosen at least `threshold`% of the time.
    
    Returns:
      - A dictionary mapping scenario_id to a dict with keys "action1" and "action2",
        where each key maps to a list of tuples (model_id, percentage).
      - A dictionary mapping scenario_id to a representative question_text.
    """
    strong_preferences = {}
    scenario_question_text = {}

    # Save one representative question text per scenario (using the first occurrence)
    for scenario_id, group in df.groupby("scenario_id"):
        scenario_question_text[scenario_id] = group["question_text"].iloc[0]

    # Group by scenario_id and model_id and compute decision percentages.
    grouped = df.groupby(["scenario_id", "model_id"])
    for (scenario_id, model_id), group in grouped:
        total = len(group)
        # Use value_counts with normalization to get percentages.
        decision_percentages = group["decision"].value_counts(normalize=True) * 100

        # Check if either decision meets the strong preference threshold.
        for decision in ["action1", "action2"]:
            perc = decision_percentages.get(decision, 0)
            if perc >= threshold:
                if scenario_id not in strong_preferences:
                    strong_preferences[scenario_id] = {"action1": [], "action2": []}
                # Round percentage to two decimal places for clarity.
                strong_preferences[scenario_id][decision].append((model_id, round(perc, 2)))
                # Since a model can only strongly prefer one action (the percentages sum to 100),
                # we break after finding a strong preference.
                break

    return strong_preferences, scenario_question_text


def filter_disagreements(strong_preferences):
    """
    Filter for scenario_ids where:
      - At least two models have a strong preference overall.
      - At least one model strongly preferred "action1" AND at least one model strongly preferred "action2".
      
    Returns a dictionary of scenarios that satisfy the above conditions.
    """
    disagreements = {}
    # First, consider only scenarios with at least two models having a strong preference.
    for scenario_id, prefs in strong_preferences.items():
        total_strong = len(prefs.get("action1", [])) + len(prefs.get("action2", []))
        if total_strong < 2:
            continue
        # Check that there is at least one model with a strong preference for each action.
        if prefs.get("action1") and prefs.get("action2"):
            disagreements[scenario_id] = prefs
    return disagreements


def save_output(disagreements, scenario_question_text, output_filename="moral_disagreements_analysis.csv"):
    """
    Save the disagreement analysis to a CSV file. Each row contains:
      - scenario_id
      - question_text (from any instance of that scenario)
      - List of models with strong preferences for "action1" (with percentages)
      - List of models with strong preferences for "action2" (with percentages)
    """
    output_rows = []
    for scenario_id, prefs in disagreements.items():
        row = {
            "scenario_id": scenario_id,
            "question_text": scenario_question_text.get(scenario_id, ""),
            "models_strong_action1": "; ".join([f"{model} ({perc}%)" for model, perc in prefs.get("action1", [])]),
            "models_strong_action2": "; ".join([f"{model} ({perc}%)" for model, perc in prefs.get("action2", [])])
        }
        output_rows.append(row)

    output_df = pd.DataFrame(
        output_rows,
        columns=["scenario_id", "question_text", "models_strong_action1", "models_strong_action2"]
    )

    try:
        output_df.to_csv(output_filename, index=False)
        print(f"[INFO] Output file '{output_filename}' created successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to write output CSV file: {e}")


def main():
    # Parse command-line arguments.
    parser = argparse.ArgumentParser(
        description="Analyze moral decision-making patterns across different LLMs."
    )
    parser.add_argument(
        "folder",
        type=str,
        nargs='?',
        default="C:/Users/jacks/git/moralchoice/data/responses/paper/high",
        help="Path to the folder containing CSV files with moral decision data."
    )
    args = parser.parse_args()
    folder_path = args.folder

    # Step 1: Load and combine CSV files.
    combined_df, files_processed = load_csv_files(folder_path)
    if combined_df is None:
        return

    print(f"[INFO] Number of CSV files processed: {files_processed}")
    total_scenarios = combined_df["scenario_id"].nunique()
    print(f"[INFO] Total number of unique scenarios: {total_scenarios}")

    # Step 2: For each (scenario_id, model_id), calculate decision percentages and flag strong preferences.
    strong_preferences, scenario_question_text = compute_strong_preferences(combined_df, threshold=75)

    # Count scenarios with at least two models having strong preferences.
    scenarios_with_two_models = {
        sc: prefs
        for sc, prefs in strong_preferences.items()
        if (len(prefs.get("action1", [])) + len(prefs.get("action2", []))) >= 2
    }
    print(f"[INFO] Number of scenarios where at least two models had strong preferences: {len(scenarios_with_two_models)}")

    # Step 3: Filter for scenarios with disagreements (i.e., at least one strong action1 and one strong action2).
    disagreements = filter_disagreements(strong_preferences)
    print(f"[INFO] Final number of scenarios with disagreement: {len(disagreements)}")

    # Print detailed preference information for each identified scenario.
    for scenario_id, prefs in disagreements.items():
        strong_a1 = "; ".join([f"{model} ({perc}%)" for model, perc in prefs.get("action1", [])])
        strong_a2 = "; ".join([f"{model} ({perc}%)" for model, perc in prefs.get("action2", [])])
        print(f"Scenario {scenario_id}:")
        print(f"  Models with strong action1: {strong_a1}")
        print(f"  Models with strong action2: {strong_a2}")

    # Step 4: Save the analysis to a CSV file.
    save_output(disagreements, scenario_question_text, output_filename="moral_disagreements_analysis.csv")


if __name__ == "__main__":
    main()
