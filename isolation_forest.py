#!/usr/bin/env python3
"""
Isolation Forest Anomaly Detection for Git Repositories
========================================================

Universal anomaly detector that analyzes git commits using Isolation Forest
machine learning algorithm with comprehensive feature engineering.

Usage:
    python3 isolation_forest.py <repo_name> <num_commits>

Example:
    python3 isolation_forest.py pandas 1000
    python3 isolation_forest.py xz 500

Features:
- Analyzes commit metrics (size, complexity, composition)
- Detects binary files and suspicious extensions
- Identifies test binaries (supply chain attack detection)
- Calculates risk scores and temporal patterns
- Uses Isolation Forest for unsupervised anomaly detection
- Generates comprehensive reports

Author: Git Anomaly Detection System
Date: 2025-11-21
"""

import os
import sys
import json
import subprocess
import re
from datetime import datetime
from collections import Counter, defaultdict
from typing import Dict, List, Set, Optional
import warnings
warnings.filterwarnings('ignore')


def run_git_command(repo_path: str, command: List[str]) -> str:
    """Execute a git command and return output."""
    try:
        result = subprocess.run(
            command,
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
            timeout=60
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Git command failed: {' '.join(command)}")
        print(f"Error: {e.stderr}")
        return ""
    except subprocess.TimeoutExpired:
        print(f"Command timed out: {' '.join(command)}")
        return ""


def get_commit_list(repo_path: str, num_commits: int) -> List[str]:
    """Get list of commit hashes."""
    print(f"Fetching last {num_commits} commits...")
    output = run_git_command(repo_path, [
        'git', 'log', '--no-merges', f'-n', str(num_commits),
        '--format=%H'
    ])
    
    commits = [h.strip() for h in output.strip().split('\n') if h.strip()]
    print(f"Found {len(commits)} commits")
    return commits


def extract_commit_features(repo_path: str, commit_hash: str) -> Optional[Dict]:
    """Extract comprehensive features for a single commit."""
    
    # Get commit metadata
    metadata_output = run_git_command(repo_path, [
        'git', 'show', '-s',
        '--format=%H|||%an|||%ae|||%ci|||%s',
        commit_hash
    ])
    
    if not metadata_output:
        return None
    
    parts = metadata_output.strip().split('|||')
    if len(parts) != 5:
        return None
    
    full_hash, author_name, author_email, date_str, subject = parts
    
    # Parse date
    try:
        date_obj = datetime.fromisoformat(date_str.replace(' ', 'T').rsplit(' ', 1)[0])
        hour = date_obj.hour
        is_weekend = date_obj.weekday() >= 5
    except:
        hour = 0
        is_weekend = False
    
    # Get diff statistics
    stat_output = run_git_command(repo_path, [
        'git', 'show', '--numstat', '--format=', commit_hash
    ])
    
    lines_added = 0
    lines_deleted = 0
    files_changed = []
    extensions = []
    binary_files = []
    test_binary_files = []
    suspicious_extensions = []
    
    # Suspicious file patterns
    suspicious_patterns = {
        'compressed': ['.xz', '.gz', '.bz2', '.zip', '.tar', '.7z', '.rar'],
        'binary': ['.bin', '.o', '.so', '.dll', '.exe', '.dylib', '.a'],
        'media': ['.png', '.jpg', '.gif', '.pdf', '.mp3', '.mp4'],
        'data': ['.dat', '.db', '.sqlite']
    }
    
    for line in stat_output.strip().split('\n'):
        if not line:
            continue
        
        parts = line.split('\t')
        if len(parts) != 3:
            continue
        
        added, deleted, filepath = parts
        
        # Handle binary files (git shows '-')
        is_binary = (added == '-' and deleted == '-')
        
        if not is_binary:
            try:
                lines_added += int(added)
                lines_deleted += int(deleted)
            except:
                pass
        
        files_changed.append(filepath)
        
        # Extract extension
        if '.' in filepath:
            ext = os.path.splitext(filepath)[1].lower()
            extensions.append(ext)
            
            # Check for suspicious extensions
            for category, ext_list in suspicious_patterns.items():
                if ext in ext_list:
                    suspicious_extensions.append((filepath, ext, category))
        
        # Track binary files
        if is_binary:
            binary_files.append(filepath)
            if 'test' in filepath.lower():
                test_binary_files.append(filepath)
    
    # Calculate subsystems (top-level directories)
    subsystems = set()
    for filepath in files_changed:
        parts = filepath.split('/')
        if len(parts) > 1:
            subsystems.add(parts[0])
        else:
            subsystems.add('root')
    
    # Categorize files
    code_files = 0
    test_files = 0
    doc_files = 0
    config_files = 0
    
    code_extensions = ['.c', '.h', '.cpp', '.hpp', '.py', '.js', '.ts', '.java', '.go', '.rs', '.rb', '.php']
    doc_extensions = ['.md', '.txt', '.rst', '.html', '.xml', '.adoc']
    config_extensions = ['.json', '.yml', '.yaml', '.cfg', '.conf', '.ini', '.toml']
    
    for filepath in files_changed:
        lower = filepath.lower()
        
        if 'test' in lower or 'spec' in lower:
            test_files += 1
        elif any(lower.endswith(x) for x in doc_extensions):
            doc_files += 1
        elif any(lower.endswith(x) for x in config_extensions):
            config_files += 1
        elif any(lower.endswith(x) for x in code_extensions):
            code_files += 1
    
    # Check for critical files
    critical_patterns = ['crypto', 'ssl', 'tls', 'auth', 'security', 'password', 'key', 'cert']
    critical_files = sum(1 for f in files_changed if any(p in f.lower() for p in critical_patterns))
    
    # Calculate metrics
    churn = lines_added + lines_deleted
    num_files = len(files_changed)
    num_subsystems = len(subsystems)
    
    # Advanced features
    subsystem_diversity = num_subsystems / max(num_files, 1)
    lines_per_file = churn / max(num_files, 1)
    
    code_ratio = code_files / max(num_files, 1)
    test_ratio = test_files / max(num_files, 1)
    
    add_del_ratio = lines_added / max(lines_deleted, 1)
    net_lines = lines_added - lines_deleted
    
    binary_ratio = len(binary_files) / max(num_files, 1)
    test_binary_ratio = len(test_binary_files) / max(num_files, 1)
    
    num_unique_extensions = len(set(extensions))
    extension_diversity = num_unique_extensions / max(len(extensions), 1)
    
    # Risk score
    risk_score = (
        critical_files * 1.5 +
        num_subsystems * 1.0 +
        is_weekend * 2.0 +
        (1 if hour < 6 or hour > 22 else 0) * 2.0
    )
    
    # Noise detection (non-code commits)
    is_noise_only = int(code_files == 0 and test_files == 0)
    
    return {
        'hash': full_hash,
        'author_name': author_name,
        'author_email': author_email,
        'date': date_str,
        'subject': subject,
        'hour': hour,
        'is_weekend': int(is_weekend),
        
        # Size metrics
        'lines_added': lines_added,
        'lines_deleted': lines_deleted,
        'churn': churn,
        'net_lines': net_lines,
        'num_files_changed': num_files,
        
        # Complexity metrics
        'num_subsystems_touched': num_subsystems,
        'subsystem_diversity': subsystem_diversity,
        'lines_per_file': lines_per_file,
        
        # Composition metrics
        'num_code_files': code_files,
        'num_test_files': test_files,
        'num_doc_files': doc_files,
        'num_config_files': config_files,
        'code_ratio': code_ratio,
        'test_ratio': test_ratio,
        
        # Binary and extension metrics
        'num_binary_files': len(binary_files),
        'num_test_binary_files': len(test_binary_files),
        'binary_ratio': binary_ratio,
        'test_binary_ratio': test_binary_ratio,
        'num_unique_extensions': num_unique_extensions,
        'extension_diversity': extension_diversity,
        'num_suspicious_extensions': len(suspicious_extensions),
        'has_compressed_files': int(any(ext in ['.xz', '.gz', '.bz2', '.zip', '.tar'] for ext in extensions)),
        'has_test_binaries': int(len(test_binary_files) > 0),
        
        # Risk metrics
        'num_critical_files': critical_files,
        'has_critical_files': int(critical_files > 0),
        'risk_score': risk_score,
        
        # Other
        'is_noise_only': is_noise_only,
        'add_del_ratio': add_del_ratio,
        
        # Raw data
        'files_changed': files_changed,
        'subsystems': sorted(list(subsystems)),
        'test_binary_files': test_binary_files,
        'suspicious_extensions': suspicious_extensions
    }


def extract_all_commits(repo_path: str, commits: List[str]) -> List[Dict]:
    """Extract features for all commits."""
    all_features = []
    total = len(commits)
    
    print(f"\nExtracting features from {total} commits...")
    
    for i, commit_hash in enumerate(commits, 1):
        if i % 50 == 0 or i == total:
            print(f"Progress: {i}/{total} ({i/total*100:.1f}%)")
        
        features = extract_commit_features(repo_path, commit_hash)
        if features:
            all_features.append(features)
    
    print(f"Successfully extracted {len(all_features)} commits")
    return all_features


def run_isolation_forest(data: List[Dict], contamination: float = 0.01) -> Dict:
    """Run Isolation Forest anomaly detection."""
    
    try:
        import pandas as pd
        import numpy as np
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("ERROR: Required packages not installed.")
        print("Please install: pip install pandas numpy scikit-learn")
        sys.exit(1)
    
    print(f"\n{'='*80}")
    print("ISOLATION FOREST ANOMALY DETECTION")
    print(f"{'='*80}")
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Select features for modeling
    feature_cols = [
        # Size metrics
        'lines_added', 'lines_deleted', 'churn', 'num_files_changed', 'net_lines',
        
        # Complexity metrics
        'num_subsystems_touched', 'subsystem_diversity', 'lines_per_file', 'num_code_files',
        
        # Composition metrics
        'code_ratio', 'test_ratio', 'num_test_files', 'num_doc_files', 'num_config_files',
        
        # Binary/Extension metrics (CRITICAL for supply chain attacks)
        'num_binary_files', 'num_test_binary_files', 'binary_ratio', 'test_binary_ratio',
        'num_unique_extensions', 'extension_diversity', 'num_suspicious_extensions',
        'has_compressed_files', 'has_test_binaries',
        
        # Risk metrics
        'has_critical_files', 'num_critical_files', 'risk_score',
        
        # Temporal metrics
        'hour', 'is_weekend',
        
        # Other
        'is_noise_only', 'add_del_ratio'
    ]
    
    print(f"\nFeatures used: {len(feature_cols)}")
    print(f"Contamination rate: {contamination} ({contamination*100:.1f}%)")
    
    # Prepare feature matrix
    X = df[feature_cols].fillna(0).values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train Isolation Forest
    print("\nTraining Isolation Forest model...")
    iso_forest = IsolationForest(
        contamination=contamination,
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    
    predictions = iso_forest.fit_predict(X_scaled)
    scores = iso_forest.score_samples(X_scaled)
    
    # Add results to dataframe
    df['anomaly_score'] = scores
    df['is_anomaly'] = predictions == -1
    
    # Statistics
    num_anomalies = (predictions == -1).sum()
    score_range = (scores.min(), scores.max())
    
    print(f"\nResults:")
    print(f"  Total commits: {len(df)}")
    print(f"  Anomalies detected: {num_anomalies} ({num_anomalies/len(df)*100:.1f}%)")
    print(f"  Score range: [{score_range[0]:.4f}, {score_range[1]:.4f}]")
    
    return {
        'dataframe': df,
        'feature_cols': feature_cols,
        'num_anomalies': int(num_anomalies),
        'score_range': score_range,
        'contamination': contamination
    }


def generate_report(results: Dict, repo_name: str, output_file: str):
    """Generate comprehensive anomaly detection report."""
    
    df = results['dataframe']
    feature_cols = results['feature_cols']
    
    # Get anomalies sorted by score
    anomalies = df[df['is_anomaly']].sort_values('anomaly_score')
    top_20 = df.nsmallest(20, 'anomaly_score')
    
    with open(output_file, 'w') as f:
        f.write("="*100 + "\n")
        f.write("ISOLATION FOREST ANOMALY DETECTION REPORT\n")
        f.write("="*100 + "\n\n")
        
        f.write(f"Repository: {repo_name}\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Commits Analyzed: {len(df)}\n")
        f.write(f"Anomalies Detected: {results['num_anomalies']} ({results['num_anomalies']/len(df)*100:.1f}%)\n")
        f.write(f"Contamination Rate: {results['contamination']*100:.1f}%\n")
        f.write(f"Features Used: {len(feature_cols)}\n\n")
        
        f.write("="*100 + "\n")
        f.write("MODEL CONFIGURATION\n")
        f.write("="*100 + "\n\n")
        
        f.write("Feature Categories:\n")
        f.write("  Size Metrics (5): lines_added, lines_deleted, churn, num_files_changed, net_lines\n")
        f.write("  Complexity (4): num_subsystems_touched, subsystem_diversity, lines_per_file, num_code_files\n")
        f.write("  Composition (5): code_ratio, test_ratio, num_test_files, num_doc_files, num_config_files\n")
        f.write("  Binary/Extension (9): num_binary_files, num_test_binary_files, binary_ratio, test_binary_ratio,\n")
        f.write("                        num_unique_extensions, extension_diversity, num_suspicious_extensions,\n")
        f.write("                        has_compressed_files, has_test_binaries\n")
        f.write("  Risk (3): has_critical_files, num_critical_files, risk_score\n")
        f.write("  Temporal (2): hour, is_weekend\n")
        f.write("  Other (3): is_noise_only, add_del_ratio\n\n")
        
        f.write("="*100 + "\n")
        f.write("TOP 20 ANOMALIES (by anomaly score)\n")
        f.write("="*100 + "\n\n")
        
        for i, (idx, row) in enumerate(top_20.iterrows(), 1):
            f.write(f"{i}. COMMIT: {row['hash'][:12]}\n")
            f.write(f"   Anomaly Score: {row['anomaly_score']:.4f}\n")
            f.write(f"   Flagged: {'YES' if row['is_anomaly'] else 'NO'}\n")
            f.write(f"   Subject: {row['subject']}\n")
            f.write(f"   Author: {row['author_name']}\n")
            f.write(f"   Date: {row['date']}\n\n")
            
            f.write(f"   Key Metrics:\n")
            f.write(f"     Churn: {row['churn']:.0f} (+{row['lines_added']:.0f}/-{row['lines_deleted']:.0f})\n")
            f.write(f"     Files: {row['num_files_changed']:.0f}, Subsystems: {row['num_subsystems_touched']:.0f}\n")
            f.write(f"     Code Files: {row['num_code_files']:.0f}, Test Files: {row['num_test_files']:.0f}\n")
            f.write(f"     Code Ratio: {row['code_ratio']:.1%}, Test Ratio: {row['test_ratio']:.1%}\n")
            
            if row['num_binary_files'] > 0:
                f.write(f"     🚨 Binary Files: {row['num_binary_files']:.0f}, Binary Ratio: {row['binary_ratio']:.1%}\n")
            
            if row['num_test_binary_files'] > 0:
                f.write(f"     🚨🚨 TEST BINARY FILES: {row['num_test_binary_files']:.0f} (HIGHLY SUSPICIOUS!)\n")
                if row['test_binary_files']:
                    for tbf in row['test_binary_files']:
                        f.write(f"          - {tbf}\n")
            
            if row['has_compressed_files']:
                f.write(f"     ⚠️  Has Compressed Files\n")
            
            if row['num_critical_files'] > 0:
                f.write(f"     Critical Files: {row['num_critical_files']:.0f}\n")
            
            f.write(f"     Risk Score: {row['risk_score']:.1f}\n")
            f.write(f"     Time: {row['hour']:.0f}:00, Weekend: {bool(row['is_weekend'])}\n")
            
            f.write("\n" + "-"*100 + "\n\n")
        
        # Statistical comparison
        f.write("="*100 + "\n")
        f.write("STATISTICAL ANALYSIS\n")
        f.write("="*100 + "\n\n")
        
        f.write("Anomalies vs Normal Commits:\n\n")
        
        anomaly_df = df[df['is_anomaly']]
        normal_df = df[~df['is_anomaly']]
        
        metrics = ['churn', 'num_files_changed', 'num_subsystems_touched', 'binary_ratio', 
                   'test_binary_ratio', 'risk_score', 'num_test_binary_files']
        
        for metric in metrics:
            f.write(f"{metric.upper()}:\n")
            f.write(f"  Anomalies - Mean: {anomaly_df[metric].mean():.2f}, Median: {anomaly_df[metric].median():.2f}, Max: {anomaly_df[metric].max():.2f}\n")
            f.write(f"  Normal    - Mean: {normal_df[metric].mean():.2f}, Median: {normal_df[metric].median():.2f}, Max: {normal_df[metric].max():.2f}\n\n")
        
        # Supply chain attack indicators
        f.write("="*100 + "\n")
        f.write("SUPPLY CHAIN ATTACK INDICATORS\n")
        f.write("="*100 + "\n\n")
        
        test_binary_commits = df[df['has_test_binaries'] == 1]
        f.write(f"Commits with test binaries: {len(test_binary_commits)} ({len(test_binary_commits)/len(df)*100:.2f}%)\n")
        
        if len(test_binary_commits) > 0:
            f.write(f"\n🚨 WARNING: Test binary files detected!\n\n")
            for idx, row in test_binary_commits.iterrows():
                f.write(f"  {row['hash'][:12]} - {row['subject']}\n")
                f.write(f"    Test Binaries: {row['num_test_binary_files']:.0f}, Binary Ratio: {row['binary_ratio']:.1%}\n")
                f.write(f"    Anomaly Score: {row['anomaly_score']:.4f}, Flagged: {row['is_anomaly']}\n")
                if row['test_binary_files']:
                    for tbf in row['test_binary_files'][:5]:  # Show first 5
                        f.write(f"      - {tbf}\n")
                f.write("\n")
        else:
            f.write("✓ No test binary files detected (good security posture)\n\n")
        
        f.write("="*100 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*100 + "\n")
    
    print(f"\nReport saved to: {output_file}")


def save_json_results(results: Dict, repo_name: str, output_file: str):
    """Save results as JSON."""
    
    df = results['dataframe']
    
    # Get top 20 anomalies
    top_20 = df.nsmallest(20, 'anomaly_score')
    
    output = {
        'repository': repo_name,
        'analysis_date': datetime.now().isoformat(),
        'total_commits': len(df),
        'num_anomalies': results['num_anomalies'],
        'contamination': results['contamination'],
        'features_used': results['feature_cols'],
        'top_20_anomalies': []
    }
    
    for idx, row in top_20.iterrows():
        commit_data = {
            'rank': int(idx + 1),
            'hash': row['hash'],
            'subject': row['subject'],
            'author': row['author_name'],
            'date': row['date'],
            'anomaly_score': float(row['anomaly_score']),
            'is_anomaly': bool(row['is_anomaly']),
            'metrics': {
                'churn': int(row['churn']),
                'lines_added': int(row['lines_added']),
                'lines_deleted': int(row['lines_deleted']),
                'num_files_changed': int(row['num_files_changed']),
                'num_subsystems_touched': int(row['num_subsystems_touched']),
                'binary_ratio': float(row['binary_ratio']),
                'test_binary_ratio': float(row['test_binary_ratio']),
                'num_test_binary_files': int(row['num_test_binary_files']),
                'has_test_binaries': bool(row['has_test_binaries']),
                'risk_score': float(row['risk_score'])
            }
        }
        
        if row['test_binary_files']:
            commit_data['test_binary_files'] = row['test_binary_files']
        
        output['top_20_anomalies'].append(commit_data)
    
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"JSON results saved to: {output_file}")


def main():
    """Main execution."""
    
    if len(sys.argv) != 3:
        print("Usage: python3 isolation_forest.py <repo_name> <num_commits>")
        print("\nExample:")
        print("  python3 isolation_forest.py pandas 1000")
        print("  python3 isolation_forest.py xz 500")
        sys.exit(1)
    
    repo_name = sys.argv[1]
    try:
        num_commits = int(sys.argv[2])
    except ValueError:
        print(f"Error: num_commits must be an integer, got '{sys.argv[2]}'")
        sys.exit(1)
    
    repo_path = os.path.join(os.getcwd(), repo_name)
    
    # Validate repository
    if not os.path.exists(repo_path):
        print(f"Error: Repository '{repo_name}' not found at {repo_path}")
        sys.exit(1)
    
    if not os.path.exists(os.path.join(repo_path, '.git')):
        print(f"Error: '{repo_name}' is not a git repository")
        sys.exit(1)
    
    print("="*80)
    print("ISOLATION FOREST ANOMALY DETECTION")
    print("="*80)
    print(f"\nRepository: {repo_name}")
    print(f"Path: {repo_path}")
    print(f"Commits to analyze: {num_commits}")
    
    # Get commits
    commits = get_commit_list(repo_path, num_commits)
    
    if not commits:
        print("Error: No commits found")
        sys.exit(1)
    
    # Extract features
    commit_data = extract_all_commits(repo_path, commits)
    
    if not commit_data:
        print("Error: Failed to extract commit data")
        sys.exit(1)
    
    # Run anomaly detection
    results = run_isolation_forest(commit_data, contamination=0.01)
    
    # Generate outputs
    output_prefix = f"{repo_name}_isolation_forest"
    report_file = f"{output_prefix}_report.txt"
    json_file = f"{output_prefix}_results.json"
    
    generate_report(results, repo_name, report_file)
    save_json_results(results, repo_name, json_file)
    
    # Summary
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"\nRepository: {repo_name}")
    print(f"Commits Analyzed: {len(commit_data)}")
    print(f"Anomalies Detected: {results['num_anomalies']} ({results['num_anomalies']/len(commit_data)*100:.1f}%)")
    print(f"\nOutputs:")
    print(f"  - Text Report: {report_file}")
    print(f"  - JSON Results: {json_file}")
    
    # Highlight supply chain risks
    df = results['dataframe']
    test_binary_count = (df['has_test_binaries'] == 1).sum()
    
    if test_binary_count > 0:
        print(f"\n🚨 SECURITY WARNING:")
        print(f"  Found {test_binary_count} commit(s) with test binary files!")
        print(f"  This may indicate supply chain attack attempts.")
        print(f"  Review the report for details.")
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
