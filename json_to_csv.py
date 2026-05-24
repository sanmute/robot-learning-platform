#!/usr/bin/env python3
"""
Convert experiment JSON files to master results spreadsheet for statistical analysis.

This script:
1. Reads all exp_*_results*.json files from a directory
2. Extracts raw trial data from each
3. Combines into one master CSV with all trials
4. Generates summary statistics table (ready for papers)

Usage:
    python json_to_csv.py --input-dir /path/to/json/files --output-dir /path/to/output
"""

import json
import glob
import pandas as pd
import numpy as np
from scipy import stats
import argparse
from pathlib import Path

def extract_trials_from_json(json_file):
    """Extract raw trial data from experiment JSON file."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    trials = []
    if 'trials' in data:
        for trial in data['trials']:
            trial_record = {
                'experiment': trial.get('experiment'),
                'condition': trial.get('condition'),
                'objective': trial.get('testObjective', trial.get('objective')),
                'score': trial.get('results', {}).get('objectiveScore'),
                'food': trial.get('results', {}).get('foodCollected'),
                'wall_bounces': trial.get('results', {}).get('wallBounces'),
                'ltm_usage': trial.get('results', {}).get('ltmUsageRate'),
            }
            
            # Add optional fields if present
            if 'checkpoint' in trial:
                trial_record['checkpoint'] = trial['checkpoint']
            if 'rep' in trial:
                trial_record['rep'] = trial['rep']
            if 'trial' in trial:
                trial_record['trial'] = trial['trial']
            if 'weightConfig' in trial:
                trial_record['weight_config'] = trial['weightConfig']
            if 'domain' in trial:
                trial_record['domain'] = trial['domain']
            
            trials.append(trial_record)
    
    return pd.DataFrame(trials)

def process_all_experiments(input_dir):
    """Process all experiment JSON files and combine into master dataframe."""
    
    json_files = sorted(glob.glob(str(Path(input_dir) / 'exp*_results*.json')))
    
    if not json_files:
        print(f"❌ No experiment JSON files found in {input_dir}")
        return None
    
    print(f"✅ Found {len(json_files)} experiment files")
    
    all_trials = []
    for json_file in json_files:
        print(f"  Reading {Path(json_file).name}...", end=' ')
        try:
            df = extract_trials_from_json(json_file)
            all_trials.append(df)
            print(f"({len(df)} trials)")
        except Exception as e:
            print(f"❌ ERROR: {e}")
    
    # Combine all
    master_df = pd.concat(all_trials, ignore_index=True)
    
    # Clean up and ensure numeric types
    master_df = master_df.dropna(subset=['score', 'condition'])
    master_df['score'] = pd.to_numeric(master_df['score'], errors='coerce')
    master_df['food'] = pd.to_numeric(master_df['food'], errors='coerce')
    master_df['wall_bounces'] = pd.to_numeric(master_df['wall_bounces'], errors='coerce')
    master_df['ltm_usage'] = pd.to_numeric(master_df['ltm_usage'], errors='coerce')
    
    return master_df

def _exp_sort_key(v):
    """Sort key that orders experiment identifiers numerically where possible,
    falling back to string order for values like '5.5.5'."""
    try:
        return (0, float(str(v)))
    except ValueError:
        return (1, str(v))

def generate_summary_statistics(master_df):
    """Generate summary statistics table for each experiment/objective/condition."""

    summary_records = []

    # Group by experiment and objective
    for exp_num in sorted(master_df['experiment'].unique(), key=_exp_sort_key):
        for objective in sorted(master_df[master_df['experiment'] == exp_num]['objective'].unique()):
            for condition in ['A', 'D']:
                # Get data
                subset = master_df[
                    (master_df['experiment'] == exp_num) &
                    (master_df['objective'] == objective) &
                    (master_df['condition'] == condition)
                ]['score']
                
                if len(subset) == 0:
                    continue
                
                summary_records.append({
                    'Experiment': f'Exp {exp_num}',
                    'Objective': objective,
                    'Condition': condition,
                    'N': len(subset),
                    'Mean': f"{subset.mean():.2f}",
                    'Std': f"{subset.std():.2f}",
                    'SEM': f"{subset.sem():.2f}",
                    'Min': f"{subset.min():.1f}",
                    'Max': f"{subset.max():.1f}",
                    'CI_Low': f"{subset.mean() - 1.96*subset.sem():.2f}",
                    'CI_High': f"{subset.mean() + 1.96*subset.sem():.2f}",
                })
    
    return pd.DataFrame(summary_records)

def generate_statistical_tests(master_df):
    """Generate t-tests comparing Condition D vs Condition A."""
    
    test_records = []
    
    # Group by experiment and objective
    for exp_num in sorted(master_df['experiment'].unique(), key=_exp_sort_key):
        for objective in sorted(master_df[master_df['experiment'] == exp_num]['objective'].unique()):
            # Get condition A and D
            cond_a = master_df[
                (master_df['experiment'] == exp_num) & 
                (master_df['objective'] == objective) &
                (master_df['condition'] == 'A')
            ]['score']
            
            cond_d = master_df[
                (master_df['experiment'] == exp_num) & 
                (master_df['objective'] == objective) &
                (master_df['condition'] == 'D')
            ]['score']
            
            if len(cond_a) == 0 or len(cond_d) == 0:
                continue
            
            # T-test
            t_stat, p_val = stats.ttest_ind(cond_d, cond_a)
            
            # Cohen's d
            pooled_std = np.sqrt((cond_a.std()**2 + cond_d.std()**2) / 2)
            cohens_d = (cond_d.mean() - cond_a.mean()) / pooled_std if pooled_std > 0 else 0
            
            # Advantage
            advantage_pct = (cond_d.mean() - cond_a.mean()) / cond_a.mean() * 100 if cond_a.mean() != 0 else 0
            
            # Significance level
            if p_val < 0.001:
                sig = "***"
            elif p_val < 0.01:
                sig = "**"
            elif p_val < 0.05:
                sig = "*"
            else:
                sig = "ns"
            
            test_records.append({
                'Experiment': f'Exp {exp_num}',
                'Objective': objective,
                'Mean_A': f"{cond_a.mean():.2f}",
                'Mean_D': f"{cond_d.mean():.2f}",
                'Difference': f"{cond_d.mean() - cond_a.mean():.2f}",
                'Advantage_%': f"{advantage_pct:+.1f}%",
                't_statistic': f"{t_stat:.3f}",
                'p_value': f"{p_val:.4f}",
                'Cohen_d': f"{cohens_d:.3f}",
                'Significance': sig,
                'N_A': len(cond_a),
                'N_D': len(cond_d),
            })
    
    return pd.DataFrame(test_records)

def main():
    parser = argparse.ArgumentParser(
        description='Convert experiment JSON files to master results CSV'
    )
    parser.add_argument(
        '--input-dir',
        default='/mnt/user-data/uploads',
        help='Directory containing experiment JSON files'
    )
    parser.add_argument(
        '--output-dir',
        default='/mnt/user-data/outputs',
        help='Directory to write output CSV files'
    )
    
    args = parser.parse_args()
    
    # Process all experiments
    print("\n" + "="*70)
    print("EXTRACTING DATA FROM EXPERIMENT JSON FILES")
    print("="*70)
    
    master_df = process_all_experiments(args.input_dir)
    
    if master_df is None:
        return
    
    print(f"\n✅ Total trials extracted: {len(master_df)}")
    print(f"   Experiments: {master_df['experiment'].nunique()}")
    print(f"   Objectives: {master_df['objective'].nunique()}")
    print(f"   Conditions: {master_df['condition'].nunique()}")
    
    # Save master spreadsheet
    output_path = Path(args.output_dir) / 'MASTER_RESULTS_RAW_TRIALS.csv'
    master_df.to_csv(output_path, index=False)
    print(f"\n✅ Saved: {output_path}")
    
    # Generate summary statistics
    print("\n" + "="*70)
    print("GENERATING SUMMARY STATISTICS")
    print("="*70)
    
    summary_df = generate_summary_statistics(master_df)
    summary_path = Path(args.output_dir) / 'SUMMARY_STATISTICS.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"\n✅ Saved summary statistics: {summary_path}")
    print(f"   {len(summary_df)} rows (experiment × objective × condition combinations)")
    
    # Generate statistical tests
    print("\n" + "="*70)
    print("GENERATING STATISTICAL TEST RESULTS")
    print("="*70)
    
    tests_df = generate_statistical_tests(master_df)
    tests_path = Path(args.output_dir) / 'STATISTICAL_TESTS_A_VS_D.csv'
    tests_df.to_csv(tests_path, index=False)
    print(f"\n✅ Saved statistical tests: {tests_path}")
    print(f"   {len(tests_df)} rows (experiment × objective comparisons)")
    
    # Print sample of results
    print("\n" + "="*70)
    print("SAMPLE RESULTS (Statistical Tests)")
    print("="*70)
    print(tests_df.head(10).to_string())
    
    # Calculate overall statistics
    print("\n" + "="*70)
    print("OVERALL STATISTICS ACROSS ALL EXPERIMENTS")
    print("="*70)
    
    cond_a_all = master_df[master_df['condition'] == 'A']['score']
    cond_d_all = master_df[master_df['condition'] == 'D']['score']
    
    t_stat, p_val = stats.ttest_ind(cond_d_all, cond_a_all)
    pooled_std = np.sqrt((cond_a_all.std()**2 + cond_d_all.std()**2) / 2)
    cohens_d = (cond_d_all.mean() - cond_a_all.mean()) / pooled_std
    overall_advantage = (cond_d_all.mean() - cond_a_all.mean()) / cond_a_all.mean() * 100
    
    print(f"\nCondition A (n={len(cond_a_all)}): {cond_a_all.mean():.2f} ± {cond_a_all.std():.2f}")
    print(f"Condition D (n={len(cond_d_all)}): {cond_d_all.mean():.2f} ± {cond_d_all.std():.2f}")
    print(f"\nDifference: {cond_d_all.mean() - cond_a_all.mean():.2f} ({overall_advantage:+.1f}%)")
    print(f"t-test: t({len(cond_a_all) + len(cond_d_all) - 2}) = {t_stat:.3f}, p = {p_val:.6f}")
    print(f"Cohen's d: {cohens_d:.3f}")
    print(f"95% CI: [{cond_d_all.mean() - 1.96*(cond_d_all.sem() + cond_a_all.sem()/2):.2f}, "
          f"{cond_d_all.mean() + 1.96*(cond_d_all.sem() + cond_a_all.sem()/2):.2f}]")
    
    if p_val < 0.05:
        print(f"✅ STATISTICALLY SIGNIFICANT (p < 0.05)")
    else:
        print(f"⚠️ NOT STATISTICALLY SIGNIFICANT (p >= 0.05)")
    
    print("\n" + "="*70)
    print("✅ COMPLETE! Ready for academic papers")
    print("="*70)
    print(f"\nNext steps:")
    print(f"1. Use SUMMARY_STATISTICS.csv → Build Table 1 for papers")
    print(f"2. Use STATISTICAL_TESTS_A_VS_D.csv → Report p-values & effect sizes")
    print(f"3. Use MASTER_RESULTS_RAW_TRIALS.csv → Create figures/visualizations")

if __name__ == '__main__':
    main()