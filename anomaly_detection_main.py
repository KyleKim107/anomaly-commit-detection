

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import subprocess
import re
import math
import warnings
from datetime import datetime
import os
import json
import random
import hashlib
import pickle

# Interactive plotting with plotly
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Disable tokenizers parallelism to prevent deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Optional: Semantic Analysis
try:
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False

warnings.filterwarnings('ignore')


# ==========================================
# CACHING SYSTEM
# ==========================================

CACHE_DIR = ".cache"

def _ensure_cache_dir():
    """Create cache directory if it doesn't exist."""
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

def _get_cache_key(repo_path, commit_hash, cache_type):
    """Generate a unique cache key for a repo+commit+type combination."""
    # Normalize repo path
    repo_name = os.path.basename(os.path.abspath(repo_path))
    key_str = f"{repo_name}_{commit_hash[:12]}_{cache_type}"
    return hashlib.md5(key_str.encode()).hexdigest()

def save_to_cache(data, repo_path, commit_hash, cache_type):
    """
    Save data to cache.
    
    Args:
        data: Data to cache (must be pickle-serializable)
        repo_path: Path to the repository
        commit_hash: Starting commit hash
        cache_type: Type of cache ('embeddings', 'parsed_log', 'diffs')
    """
    _ensure_cache_dir()
    cache_key = _get_cache_key(repo_path, commit_hash, cache_type)
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.pkl")
    
    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)
    
    return cache_file

def load_from_cache(repo_path, commit_hash, cache_type):
    """
    Load data from cache if available.
    
    Args:
        repo_path: Path to the repository
        commit_hash: Starting commit hash
        cache_type: Type of cache ('embeddings', 'parsed_log', 'diffs')
        
    Returns:
        Cached data or None if not found
    """
    _ensure_cache_dir()
    cache_key = _get_cache_key(repo_path, commit_hash, cache_type)
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.pkl")
    
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Cache load error: {e}")
            return None
    return None

def is_cached(repo_path, commit_hash, cache_type):
    """Check if data is cached."""
    _ensure_cache_dir()
    cache_key = _get_cache_key(repo_path, commit_hash, cache_type)
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.pkl")
    return os.path.exists(cache_file)

def clear_cache(cache_type=None):
    """
    Clear cache files.
    
    Args:
        cache_type: If provided, only clear caches of this type. Otherwise clear all.
    """
    if not os.path.exists(CACHE_DIR):
        return
    
    for f in os.listdir(CACHE_DIR):
        if f.endswith('.pkl'):
            if cache_type is None or cache_type in f:
                os.remove(os.path.join(CACHE_DIR, f))
    
    print(f"Cache cleared{f' for {cache_type}' if cache_type else ''}")

def get_cache_stats():
    """Get statistics about cached data."""
    if not os.path.exists(CACHE_DIR):
        return {"total_files": 0, "total_size_mb": 0}
    
    files = [f for f in os.listdir(CACHE_DIR) if f.endswith('.pkl')]
    total_size = sum(os.path.getsize(os.path.join(CACHE_DIR, f)) for f in files)
    
    return {
        "total_files": len(files),
        "total_size_mb": round(total_size / (1024 * 1024), 2)
    }


# ==========================================
# 0. REPOSITORY MANAGEMENT
# ==========================================

def ensure_repo_exists(repo_path, repo_url):
    """
    Check if repository exists, clone it if not.
    
    Args:
        repo_path: Local path where repo should exist
        repo_url: GitHub URL to clone from
    """
    if not os.path.exists(repo_path):
        print(f"Repository not found at {repo_path}")
        print(f"Cloning from {repo_url}...")
        try:
            subprocess.run(['git', 'clone', repo_url, repo_path], 
                         check=True, capture_output=True)
            print(f"✓ Successfully cloned to {repo_path}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to clone: {e.stderr.decode()}")
            return False
    else:
        print(f"✓ Repository found at {repo_path}")
    return True

# ==========================================
# 1. CORE UTILITIES
# ==========================================

def calculate_entropy(text):
    if not text: return 0
    counts = Counter(text)
    length = len(text)
    return -sum((count / length) * math.log2(count / length) for count in counts.values())

def get_default_branch(repo_path):
    """
    Auto-detect the default branch of a git repository (main, master, etc.).
    
    Args:
        repo_path: Path to the git repository
        
    Returns:
        Default branch name (e.g., 'main', 'master')
    """
    try:
        # Try to get the default branch from remote HEAD
        result = subprocess.run(
            ["git", "-C", repo_path, "symbolic-ref", "refs/remotes/origin/HEAD"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            # Output is like "refs/remotes/origin/main"
            return result.stdout.strip().split('/')[-1]
    except:
        pass
    
    # Fallback: check if common branch names exist
    for branch in ['main', 'master', 'develop', 'trunk']:
        result = subprocess.run(
            ["git", "-C", repo_path, "rev-parse", "--verify", branch],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            return branch
    
    # Last resort: use HEAD
    return "HEAD"

def dump_git_log(repo_path, start_ref=None, n_commits=None, out_file="git_log.txt", repo_url=None):
    # Check if repo exists, clone if URL is provided
    if not os.path.exists(repo_path):
        if repo_url:
            print(f"Repository not found at {repo_path}. Cloning from {repo_url}...")
            try:
                subprocess.run(['git', 'clone', repo_url, repo_path], 
                             check=True, capture_output=True)
                print(f"✓ Successfully cloned to {repo_path}")
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Failed to clone repository: {e.stderr.decode()}")
        else:
            raise FileNotFoundError(f"Repository not found at {repo_path}. Provide repo_url to clone automatically.")
    
    # Auto-detect default branch if not specified
    if start_ref is None:
        start_ref = get_default_branch(repo_path)
        print(f"Auto-detected default branch: {start_ref}")
    
    cmd = [
        "git", "-C", repo_path, "log", start_ref,
        "--no-renames", "--date=iso-strict",
        # FIXED: Added %ae (email) and %cd (committer date) back for feature extraction
        # Use %H (full hash) instead of %h for reliable commit matching
        "--pretty=format:__COMMIT__%H|%an|%ae|%ad|%cd|%s",
        "--numstat" 
    ]
    if n_commits: cmd.extend(["-n", str(n_commits)])

    with open(out_file, "wb") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, check=True)
    return out_file

def parse_git_log_file(log_path):
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    
    data = []
    current_commit = None
    
    # Security-related keywords in commit messages
    SECURITY_KEYWORDS = [
        'fix', 'bug', 'error', 'crash', 'fail', 'issue', 'problem',
        'memory', 'buffer', 'overflow', 'leak', 'null', 'pointer',
        'security', 'vulnerability', 'cve', 'patch', 'sanitize',
        'validate', 'check', 'verify', 'auth', 'permission', 'access',
        'inject', 'escape', 'encode', 'decode', 'parse', 'boundary',
        'limit', 'size', 'length', 'bounds', 'range', 'index',
        'input', 'output', 'handle', 'exception', 'throw', 'catch'
    ]
    
    # Security-sensitive file patterns  
    SECURITY_FILE_PATTERNS = [
        'auth', 'login', 'password', 'crypt', 'ssl', 'tls', 'cert',
        'token', 'session', 'cookie', 'key', 'secret', 'credential',
        'permission', 'access', 'security', 'sanitiz', 'valid',
        'parse', 'decode', 'encode', 'serial', 'socket', 'network',
        'buffer', 'memory', 'alloc', 'free', 'string', 'format'
    ]
    
    # State tracking
    stats = {
        'lines_added': 0, 'lines_deleted': 0, 
        'test_files': 0, 'src_files': 0, 'doc_files': 0,
        'sensitive_files': 0,
        'security_files': 0,  # Files matching security patterns
        'c_cpp_files': 0,     # C/C++ files (memory safety concerns)
        'config_files': 0,    # Config files (access control)
    }
    
    for line in tqdm(lines, desc="Parsing Log"):
        line = line.strip()
        
        if line.startswith("__COMMIT__"):
            # 1. Save Previous Commit
            if current_commit:
                total_files = stats['test_files'] + stats['src_files'] + stats['doc_files']
                # Avoid division by zero
                safe_total = total_files if total_files > 0 else 1
                
                current_commit.update({
                    # Map to the keys expected by the Anomaly Detector
                    'lines_inserted': stats['lines_added'],
                    'lines_deleted': stats['lines_deleted'],
                    'files_changed': total_files,
                    'file_count': total_files, # Alias for consistency
                    
                    # Security Features
                    'test_ratio': stats['test_files'] / safe_total,
                    'sensitive_ratio': stats['sensitive_files'] / safe_total,
                    'security_file_ratio': stats['security_files'] / safe_total,
                    'c_cpp_file_ratio': stats['c_cpp_files'] / safe_total,
                    'config_file_ratio': stats['config_files'] / safe_total,
                    'security_file_count': stats['security_files'],
                    'c_cpp_file_count': stats['c_cpp_files'],
                })
                data.append(current_commit)

            # 2. Start New Commit
            parts = line.split("|")
            try:
                # Parse: %h|%an|%ae|%ad|%cd|%s
                c_hash = parts[0].replace("__COMMIT__", "")
                c_author = parts[1]
                c_email = parts[2]
                c_ad_str = parts[3]
                c_cd_str = parts[4]
                c_msg = "|".join(parts[5:])

                # Dates & Latency
                a_dt = datetime.fromisoformat(c_ad_str)
                c_dt = datetime.fromisoformat(c_cd_str)
                latency = (c_dt - a_dt).total_seconds()

                # Holiday Check
                is_holiday = 1 if (a_dt.month == 12 and a_dt.day >= 24) or (a_dt.month == 1 and a_dt.day == 1) else 0
                
                # Count security keywords in commit message
                msg_lower = c_msg.lower()
                security_keyword_count = sum(1 for kw in SECURITY_KEYWORDS if kw in msg_lower)
                has_fix_keyword = 1 if any(kw in msg_lower for kw in ['fix', 'bug', 'patch', 'issue', 'error']) else 0
                has_security_keyword = 1 if any(kw in msg_lower for kw in ['security', 'vulnerability', 'cve', 'auth', 'permission']) else 0
                has_memory_keyword = 1 if any(kw in msg_lower for kw in ['memory', 'buffer', 'overflow', 'leak', 'null', 'pointer']) else 0
                
                # Is this a weekend commit? (potentially less reviewed)
                is_weekend = 1 if a_dt.weekday() >= 5 else 0
                # Is this late night? (10pm - 6am)
                is_late_night = 1 if a_dt.hour >= 22 or a_dt.hour <= 6 else 0
                
                current_commit = {
                    'hash': c_hash,
                    'author': c_author,
                    'email': c_email,
                    'author_date': a_dt,
                    'hour_of_day': a_dt.hour,   # REQUIRED key
                    'day_of_week': a_dt.weekday(),
                    'merge_latency_sec': latency, # REQUIRED key
                    'is_holiday': is_holiday,
                    'is_weekend': is_weekend,
                    'is_late_night': is_late_night,
                    'msg_length': len(c_msg),     # REQUIRED key
                    'msg_entropy': calculate_entropy(c_msg), # REQUIRED key
                    'msg_content': c_msg,
                    # Security-focused message features
                    'security_keyword_count': security_keyword_count,
                    'has_fix_keyword': has_fix_keyword,
                    'has_security_keyword': has_security_keyword,
                    'has_memory_keyword': has_memory_keyword,
                }
                # Reset stats
                stats = {'lines_added': 0, 'lines_deleted': 0, 'test_files': 0, 'src_files': 0, 'doc_files': 0, 'sensitive_files': 0, 'security_files': 0, 'c_cpp_files': 0, 'config_files': 0}
            except Exception as e:
                current_commit = None
                continue
        
        elif current_commit and line:
            # Parse numstat: "10  5   src/file.c"
            parts = line.split("\t")
            if len(parts) == 3:
                added, deleted, filepath = parts
                
                if added == '-': added = 0
                if deleted == '-': deleted = 0
                
                stats['lines_added'] += int(added)
                stats['lines_deleted'] += int(deleted)

                # Classify File Type
                if "test" in filepath.lower() or filepath.endswith(".t"):
                    stats['test_files'] += 1
                elif filepath.endswith((".c", ".h", ".cpp", ".py", ".go", ".js", ".ts")):
                    stats['src_files'] += 1
                elif filepath.endswith((".md", ".txt", ".html")):
                    stats['doc_files'] += 1
                
                if "ssl/" in filepath or "crypto/" in filepath:
                    stats['sensitive_files'] += 1
                
                # Security-sensitive file detection
                filepath_lower = filepath.lower()
                if any(pattern in filepath_lower for pattern in SECURITY_FILE_PATTERNS):
                    stats['security_files'] += 1
                
                # C/C++ files (memory safety concerns)
                if filepath.endswith(('.c', '.h', '.cpp', '.hpp', '.cc', '.cxx')):
                    stats['c_cpp_files'] += 1
                
                # Config files
                if filepath.endswith(('.xml', '.yaml', '.yml', '.json', '.conf', '.cfg', '.ini', '.properties')):
                    stats['config_files'] += 1

    # Don't forget the very last commit in the file!
    if current_commit:
        total_files = stats['test_files'] + stats['src_files'] + stats['doc_files']
        safe_total = total_files if total_files > 0 else 1
        current_commit.update({
            'lines_inserted': stats['lines_added'],
            'lines_deleted': stats['lines_deleted'],
            'files_changed': total_files,
            'file_count': total_files,
            'test_ratio': stats['test_files'] / safe_total,
            'sensitive_ratio': stats['sensitive_files'] / safe_total,
            'security_file_ratio': stats['security_files'] / safe_total,
            'c_cpp_file_ratio': stats['c_cpp_files'] / safe_total,
            'config_file_ratio': stats['config_files'] / safe_total,
            'security_file_count': stats['security_files'],
            'c_cpp_file_count': stats['c_cpp_files'],
        })
        data.append(current_commit)

    df = pd.DataFrame(data)
    
    if df.empty: return df

    # Final Feature Engineering: Churn Ratio
    df['total_change'] = df['lines_inserted'] + df['lines_deleted']
    df['churn_ratio'] = df.apply(
        lambda row: row['total_change'] / row['files_changed'] if row['files_changed'] > 0 else 0, 
        axis=1
    )
    
    return df

# ==========================================
# 2. VISUALIZATION FUNCTIONS
# ==========================================

def plot_commits_3d(df, features, highlight_commit=None, title="Commit Anomaly Visualization"):
    """
    Plot all commits as a 3D scatter plot with anomaly color coding.
    
    Args:
        df: DataFrame with commit features (output of parse_git_log_file with anomaly detection)
        features: Tuple of three feature names to use as (x, y, z) axes
        highlight_commit: Optional commit hash to highlight with a cross marker
        title: Plot title
        
    Returns:
        Plotly figure object (interactive) or matplotlib figure if plotly unavailable
    """
    if len(features) != 3:
        raise ValueError("features must be a tuple of exactly 3 feature names")
    
    x_feat, y_feat, z_feat = features
    
    # Ensure required columns exist
    for feat in features:
        if feat not in df.columns:
            raise ValueError(f"Feature '{feat}' not found in dataframe. Available: {list(df.columns)}")
    
    # Ensure anomaly column exists
    if 'is_anomaly' not in df.columns:
        raise ValueError("DataFrame must have 'is_anomaly' column. Run detect_statistical_anomalies first.")
    
    # Create a copy to avoid modifying original
    plot_df = df.copy()
    
    # Create color column based on anomaly status
    plot_df['color'] = plot_df['is_anomaly'].apply(lambda x: 'Anomaly' if x else 'Normal')
    
    # Create hover text
    plot_df['hover_text'] = plot_df.apply(
        lambda row: f"Hash: {row['hash']}<br>"
                    f"Author: {row['author']}<br>"
                    f"Message: {row['msg_content'][:50]}...<br>"
                    f"{x_feat}: {row[x_feat]:.2f}<br>"
                    f"{y_feat}: {row[y_feat]:.2f}<br>"
                    f"{z_feat}: {row[z_feat]:.2f}",
        axis=1
    )
    
    if PLOTLY_AVAILABLE:
        # Interactive 3D plot with Plotly
        color_map = {'Normal': '#2ecc71', 'Anomaly': '#e74c3c'}  # Green and Red
        
        fig = px.scatter_3d(
            plot_df,
            x=x_feat,
            y=y_feat,
            z=z_feat,
            color='color',
            color_discrete_map=color_map,
            hover_data=['hash', 'author', 'msg_content'],
            title=title,
            labels={x_feat: x_feat, y_feat: y_feat, z_feat: z_feat},
            opacity=0.7
        )
        
        # Update marker size and style
        fig.update_traces(marker=dict(size=5))
        
        # Highlight specific commit if provided
        if highlight_commit:
            highlight_row = plot_df[plot_df['hash'].str.startswith(highlight_commit)]
            if not highlight_row.empty:
                row = highlight_row.iloc[0]
                # Determine color based on anomaly status
                marker_color = '#e74c3c' if row['is_anomaly'] else '#2ecc71'
                
                fig.add_trace(go.Scatter3d(
                    x=[row[x_feat]],
                    y=[row[y_feat]],
                    z=[row[z_feat]],
                    mode='markers',
                    marker=dict(
                        size=15,
                        symbol='cross',
                        color=marker_color,
                        line=dict(width=3, color='black')
                    ),
                    name=f'Highlighted: {highlight_commit}',
                    hovertext=f"<b>HIGHLIGHTED</b><br>Hash: {row['hash']}<br>"
                              f"Author: {row['author']}<br>"
                              f"Anomaly: {row['is_anomaly']}"
                ))
        
        # Update layout for better appearance
        fig.update_layout(
            scene=dict(
                xaxis_title=x_feat,
                yaxis_title=y_feat,
                zaxis_title=z_feat,
                bgcolor='rgba(240,240,240,0.9)'
            ),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            template='plotly_white'
        )
        
        fig.show()
        return fig
    
    else:
        # Fallback to matplotlib 3D plot
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot normal commits (green)
        normal = plot_df[plot_df['is_anomaly'] == False]
        ax.scatter(
            normal[x_feat], normal[y_feat], normal[z_feat],
            c='#2ecc71', alpha=0.6, s=30, label='Normal', marker='o'
        )
        
        # Plot anomaly commits (red)
        anomalies = plot_df[plot_df['is_anomaly'] == True]
        ax.scatter(
            anomalies[x_feat], anomalies[y_feat], anomalies[z_feat],
            c='#e74c3c', alpha=0.8, s=50, label='Anomaly', marker='o'
        )
        
        # Highlight specific commit
        if highlight_commit:
            highlight_row = plot_df[plot_df['hash'].str.startswith(highlight_commit)]
            if not highlight_row.empty:
                row = highlight_row.iloc[0]
                marker_color = '#e74c3c' if row['is_anomaly'] else '#2ecc71'
                ax.scatter(
                    [row[x_feat]], [row[y_feat]], [row[z_feat]],
                    c=marker_color, s=200, marker='X', 
                    edgecolors='black', linewidths=2,
                    label=f'Highlighted: {highlight_commit}'
                )
        
        ax.set_xlabel(x_feat)
        ax.set_ylabel(y_feat)
        ax.set_zlabel(z_feat)
        ax.set_title(title)
        ax.legend()
        
        plt.tight_layout()
        plt.show()
        return fig


def plot_feature_distributions(df):
    """
    Plot distribution curves for all numeric features in the commit dataframe.
    Uses Plotly for interactive browser-based visualization.
    
    Args:
        df: DataFrame with commit features (output of parse_git_log_file)
        
    Returns:
        Plotly figure object
    """
    if not PLOTLY_AVAILABLE:
        print("⚠️ Plotly not available. Install with: pip install plotly")
        return None
    
    # Select only numeric columns for distribution plots
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Exclude certain columns that don't make sense for distribution
    exclude_cols = ['anomaly_score']
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    n_features = len(feature_cols)
    n_cols = 4
    n_rows = math.ceil(n_features / n_cols)
    
    # Create subplots
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=feature_cols,
        vertical_spacing=0.08,
        horizontal_spacing=0.05
    )
    
    has_anomaly = 'is_anomaly' in df.columns
    
    for idx, feature in enumerate(feature_cols):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        
        if has_anomaly:
            # Normal commits
            normal_data = df[df['is_anomaly'] == False][feature].dropna()
            if len(normal_data) > 1:
                fig.add_trace(
                    go.Histogram(
                        x=normal_data, 
                        name='Normal', 
                        marker_color='#2ecc71',
                        opacity=0.6,
                        showlegend=(idx == 0),
                        nbinsx=30
                    ),
                    row=row, col=col
                )
            
            # Anomaly commits
            anomaly_data = df[df['is_anomaly'] == True][feature].dropna()
            if len(anomaly_data) > 1:
                fig.add_trace(
                    go.Histogram(
                        x=anomaly_data, 
                        name='Anomaly', 
                        marker_color='#e74c3c',
                        opacity=0.6,
                        showlegend=(idx == 0),
                        nbinsx=30
                    ),
                    row=row, col=col
                )
        else:
            data = df[feature].dropna()
            if len(data) > 1:
                fig.add_trace(
                    go.Histogram(
                        x=data, 
                        name=feature, 
                        marker_color='#3498db',
                        opacity=0.6,
                        showlegend=False,
                        nbinsx=30
                    ),
                    row=row, col=col
                )
    
    fig.update_layout(
        title_text='Feature Distributions',
        title_font_size=16,
        height=250 * n_rows,
        width=1200,
        barmode='overlay',
        template='plotly_white',
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
    )
    
    fig.show()
    return fig


# ==========================================
# EXPLORATORY ANALYSIS FUNCTIONS
# ==========================================

def explore_commit_features(repo_path, repo_url=None, specific_commit=None, 
                            n_commits=500, features=None, disable_cache=False):
    """
    Perform exploratory analysis of commit features for a repository.
    
    This function is designed for understanding commit patterns without running
    anomaly detection. It visualizes the distribution and relationships of
    selected features.
    
    Args:
        repo_path: Path to the git repository
        repo_url: URL to clone from if repo doesn't exist locally
        specific_commit: Specific commit to start from (None = default branch)
        n_commits: Number of commits to analyze
        features: List of features to visualize. If None, uses default set.
                  Available features:
                  - Temporal: 'hour_of_day', 'day_of_week', 'is_weekend', 'is_late_night', 'is_holiday'
                  - Size: 'lines_inserted', 'lines_deleted', 'files_changed', 'churn_ratio', 'total_change'
                  - Message: 'msg_length', 'msg_entropy'
                  - Security: 'security_keyword_count', 'has_fix_keyword', 'has_security_keyword',
                              'has_memory_keyword', 'test_ratio', 'sensitive_ratio',
                              'security_file_ratio', 'c_cpp_file_ratio', 'config_file_ratio'
                  - Time: 'merge_latency_sec'
        disable_cache: Whether to skip cache loading
        
    Returns:
        Dictionary with:
        - 'dataframe': The parsed commit DataFrame
        - 'figures': List of generated plotly figures
        - 'stats': Summary statistics for each feature
    """
    if not PLOTLY_AVAILABLE:
        print("⚠️ Plotly is required for visualizations. Install with: pip install plotly")
        return None
    
    print("=" * 80)
    print("  COMMIT FEATURE EXPLORATORY ANALYSIS")
    print("=" * 80)
    
    # Check if repo exists locally
    if not os.path.exists(repo_path):
        if not repo_url:
            raise ValueError(
                f"Repository not found at '{repo_path}' and no repo_url provided. "
                "Please provide repo_url to clone the repository."
            )
        print(f"📥 Repository not found locally. Cloning from {repo_url}...")
        try:
            subprocess.run(['git', 'clone', repo_url, repo_path], 
                         check=True, capture_output=True)
            print(f"✓ Successfully cloned to {repo_path}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to clone repository: {e.stderr.decode()}")
    else:
        print(f"📂 Using local repository: {repo_path}")
    
    # Determine the commit to start from
    if specific_commit:
        # Verify the commit exists
        verify_result = subprocess.run(
            ['git', '-C', repo_path, 'rev-parse', '--verify', specific_commit],
            capture_output=True, text=True
        )
        if verify_result.returncode != 0:
            print(f"⚠️ Commit '{specific_commit}' not found, using default branch")
            start_ref = get_default_branch(repo_path)
        else:
            start_ref = specific_commit
            print(f"🎯 Starting from commit: {specific_commit}")
    else:
        start_ref = get_default_branch(repo_path)
        print(f"🎯 Using default branch: {start_ref}")
    
    print(f"📊 Analyzing {n_commits} commits\n")
    
    # Get parsed log
    df = get_parsed_log_cached(repo_path, start_ref, n_commits, 
                               repo_url=repo_url, disable_cache=disable_cache)
    
    if df is None or df.empty:
        print("❌ Failed to parse git log")
        return None
    
    print(f"✓ Parsed {len(df)} commits\n")
    
    # Default features if none specified
    if features is None:
        features = [
            'hour_of_day', 'day_of_week', 
            'lines_inserted', 'lines_deleted', 'files_changed',
            'msg_length', 'churn_ratio'
        ]
    
    # Filter to available features
    available_features = [f for f in features if f in df.columns]
    missing_features = [f for f in features if f not in df.columns]
    
    if missing_features:
        print(f"⚠️ Missing features (not in data): {missing_features}")
    
    if not available_features:
        print("❌ No valid features to visualize")
        return {'dataframe': df, 'figures': [], 'stats': {}}
    
    print(f"📈 Visualizing features: {available_features}\n")
    
    figures = []
    stats = {}
    
    # Calculate statistics for each feature
    for feat in available_features:
        feat_data = df[feat].dropna()
        stats[feat] = {
            'count': len(feat_data),
            'mean': feat_data.mean(),
            'std': feat_data.std(),
            'min': feat_data.min(),
            'max': feat_data.max(),
            'median': feat_data.median(),
            'q25': feat_data.quantile(0.25),
            'q75': feat_data.quantile(0.75),
        }
    
    # ================================================================
    # 1. Individual Feature Distributions
    # ================================================================
    n_features = len(available_features)
    n_cols = min(3, n_features)
    n_rows = math.ceil(n_features / n_cols)
    
    fig_dist = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=available_features,
        vertical_spacing=0.12,
        horizontal_spacing=0.08
    )
    
    for idx, feat in enumerate(available_features):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        
        feat_data = df[feat].dropna()
        
        # Use bar chart for categorical-like features
        if feat in ['hour_of_day', 'day_of_week']:
            counts = feat_data.value_counts().sort_index()
            fig_dist.add_trace(
                go.Bar(x=counts.index, y=counts.values, 
                       marker_color='#3498db', name=feat, showlegend=False),
                row=row, col=col
            )
        else:
            fig_dist.add_trace(
                go.Histogram(x=feat_data, nbinsx=30, 
                            marker_color='#3498db', name=feat, showlegend=False),
                row=row, col=col
            )
    
    fig_dist.update_layout(
        title_text=f'Feature Distributions - {os.path.basename(repo_path)}',
        height=300 * n_rows,
        width=1200,
        template='plotly_white'
    )
    
    fig_dist.show()
    figures.append(fig_dist)
    
    # ================================================================
    # 2. Temporal Patterns (if temporal features present)
    # ================================================================
    temporal_features = ['hour_of_day', 'day_of_week']
    temporal_available = [f for f in temporal_features if f in available_features]
    
    if temporal_available:
        if 'hour_of_day' in df.columns:
            # Hour of day distribution
            hour_counts = df['hour_of_day'].value_counts().sort_index()
            
            fig_hour = go.Figure()
            fig_hour.add_trace(go.Bar(
                x=hour_counts.index,
                y=hour_counts.values,
                marker_color=['#e74c3c' if (h >= 22 or h <= 6) else '#3498db' 
                              for h in hour_counts.index],
                text=hour_counts.values,
                textposition='outside'
            ))
            
            fig_hour.update_layout(
                title=f'Commits by Hour of Day - {os.path.basename(repo_path)}',
                xaxis_title='Hour (0-23)',
                yaxis_title='Number of Commits',
                template='plotly_white',
                height=400,
                width=900
            )
            fig_hour.update_xaxes(tickmode='linear', tick0=0, dtick=1)
            
            fig_hour.show()
            figures.append(fig_hour)
        
        if 'day_of_week' in df.columns:
            # Day of week distribution
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_counts = df['day_of_week'].value_counts().sort_index()
            
            fig_day = go.Figure()
            fig_day.add_trace(go.Bar(
                x=[day_names[i] for i in day_counts.index],
                y=day_counts.values,
                marker_color=['#e74c3c' if d >= 5 else '#3498db' for d in day_counts.index],
                text=day_counts.values,
                textposition='outside'
            ))
            
            fig_day.update_layout(
                title=f'Commits by Day of Week - {os.path.basename(repo_path)}',
                xaxis_title='Day',
                yaxis_title='Number of Commits',
                template='plotly_white',
                height=400,
                width=900
            )
            
            fig_day.show()
            figures.append(fig_day)
    
    # ================================================================
    # 3. Correlation Heatmap (for numeric features)
    # ================================================================
    numeric_features = [f for f in available_features 
                        if df[f].dtype in ['int64', 'float64']]
    
    if len(numeric_features) >= 2:
        corr_matrix = df[numeric_features].corr()
        
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=numeric_features,
            y=numeric_features,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig_corr.update_layout(
            title=f'Feature Correlation Matrix - {os.path.basename(repo_path)}',
            height=500,
            width=700,
            template='plotly_white'
        )
        
        fig_corr.show()
        figures.append(fig_corr)
    
    # ================================================================
    # 4. Box Plots for Comparison
    # ================================================================
    if len(numeric_features) >= 2:
        # Normalize features for comparison
        df_normalized = df[numeric_features].copy()
        for col in numeric_features:
            col_data = df_normalized[col]
            col_min, col_max = col_data.min(), col_data.max()
            if col_max > col_min:
                df_normalized[col] = (col_data - col_min) / (col_max - col_min)
        
        fig_box = go.Figure()
        for feat in numeric_features:
            fig_box.add_trace(go.Box(
                y=df_normalized[feat].dropna(),
                name=feat,
                boxmean='sd'
            ))
        
        fig_box.update_layout(
            title=f'Feature Distribution Comparison (Normalized) - {os.path.basename(repo_path)}',
            yaxis_title='Normalized Value (0-1)',
            template='plotly_white',
            height=500,
            width=1000
        )
        
        fig_box.show()
        figures.append(fig_box)
    
    # ================================================================
    # 5. Scatter Matrix for Key Features
    # ================================================================
    scatter_features = [f for f in ['lines_inserted', 'lines_deleted', 'files_changed', 'churn_ratio']
                        if f in available_features]
    
    if len(scatter_features) >= 2:
        fig_scatter = px.scatter_matrix(
            df[scatter_features].dropna(),
            dimensions=scatter_features,
            title=f'Feature Scatter Matrix - {os.path.basename(repo_path)}',
            opacity=0.5
        )
        fig_scatter.update_traces(diagonal_visible=False, marker=dict(size=4))
        fig_scatter.update_layout(height=700, width=900, template='plotly_white')
        
        fig_scatter.show()
        figures.append(fig_scatter)
    
    # ================================================================
    # Print Summary Statistics
    # ================================================================
    print("\n" + "=" * 80)
    print("  FEATURE STATISTICS")
    print("=" * 80)
    
    for feat, s in stats.items():
        print(f"\n  {feat}:")
        print(f"    Count: {s['count']:,}")
        print(f"    Mean:  {s['mean']:.2f} ± {s['std']:.2f}")
        print(f"    Range: [{s['min']:.2f}, {s['max']:.2f}]")
        print(f"    Median: {s['median']:.2f} (IQR: {s['q25']:.2f} - {s['q75']:.2f})")
    
    print("\n" + "=" * 80)
    
    # List all available features for reference
    print("\n📋 All available features in dataset:")
    all_numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"   {all_numeric}")
    
    return {
        'dataframe': df,
        'figures': figures,
        'stats': stats
    }


def visualize_isolation_forest_decision_path(model, X, feature_names, sample_idx, 
                                              max_depth=4):
    """
    Visualize how Isolation Forest arrives at its decision for a specific sample.
    
    This function finds the tree with the SHORTEST path length (the tree that 
    isolated the sample most quickly) and visualizes only that tree's decision path.
    For anomalies, shorter paths indicate the sample was easier to isolate.
    
    Args:
        model: Trained IsolationForest model
        X: Feature matrix (scaled)
        feature_names: List of feature names
        sample_idx: Index of the sample to explain
        max_depth: Maximum depth to show (default 4)
        
    Returns:
        Plotly figure showing the decision path of the most decisive tree
    """
    if not PLOTLY_AVAILABLE:
        print("⚠️ Plotly is required for visualizations")
        return None
    
    sample = X[sample_idx].reshape(1, -1)
    sample_score = model.decision_function(sample)[0]
    is_anomaly = model.predict(sample)[0] == -1
    
    # Find the tree with the shortest path length (most decisive for anomalies)
    all_trees = model.estimators_
    path_lengths = []
    
    for tree in all_trees:
        node_indicator = tree.decision_path(sample)
        path_length = node_indicator.indices.shape[0]
        path_lengths.append(path_length)
    
    # Select the tree with shortest path (quickest to isolate)
    best_tree_idx = np.argmin(path_lengths)
    best_tree = all_trees[best_tree_idx]
    shortest_path = path_lengths[best_tree_idx]
    avg_path = np.mean(path_lengths)
    
    print(f"\n📊 Analyzing {len(all_trees)} trees...")
    print(f"   Shortest path: {shortest_path} steps (Tree #{best_tree_idx + 1})")
    print(f"   Average path: {avg_path:.1f} steps")
    print(f"   This tree isolated the sample {avg_path - shortest_path:.1f} steps faster than average")
    
    # Create single figure (no subplots needed)
    fig = go.Figure()
    
    # Process the best tree
    tree = best_tree
    tree_idx = 0  # Single tree
    tree_model = tree.tree_
    
    # Trace the decision path for this sample
    node_indicator = tree.decision_path(sample)
    node_indices = node_indicator.indices
    
    # Build tree structure for visualization
    nodes = []
    edges = []
    
    # Track positions for layout
    level_counts = {}
    node_positions = {}
    
    # First pass: determine levels
    def get_node_level(node_id, level=0):
        if node_id in node_positions:
            return
        node_positions[node_id] = {'level': level}
        
        left_child = tree_model.children_left[node_id]
        right_child = tree_model.children_right[node_id]
        
        if left_child != -1:
            get_node_level(left_child, level + 1)
        if right_child != -1:
            get_node_level(right_child, level + 1)
    
    get_node_level(0)
    
    # Calculate x positions based on level
    for node_id, pos in node_positions.items():
        level = pos['level']
        if level not in level_counts:
            level_counts[level] = 0
        level_counts[level] += 1
    
    level_current = {l: 0 for l in level_counts}
    
    def assign_x_position(node_id):
        level = node_positions[node_id]['level']
        total_at_level = level_counts[level]
        current = level_current[level]
        level_current[level] += 1
        
        # Spread nodes horizontally
        x = (current + 0.5) / total_at_level
        node_positions[node_id]['x'] = x
        
        left_child = tree_model.children_left[node_id]
        right_child = tree_model.children_right[node_id]
        
        if left_child != -1:
            assign_x_position(left_child)
        if right_child != -1:
            assign_x_position(right_child)
    
    assign_x_position(0)
    
    # Build visualization data
    x_nodes = []
    y_nodes = []
    node_colors = []
    node_texts = []
    hover_texts = []
    
    x_edges = []
    y_edges = []
    
    path_set = set(node_indices)
    
    def add_node(node_id, depth=0):
        if depth > max_depth:
            return
        
        pos = node_positions.get(node_id, {'x': 0.5, 'level': depth})
        x = pos['x']
        y = -pos['level']  # Negative so root is at top
        
        x_nodes.append(x)
        y_nodes.append(y)
        
        # Determine if this node is on the path
        on_path = node_id in path_set
        is_leaf = tree_model.children_left[node_id] == -1
        
        if is_leaf:
            # Leaf node - bright green for path, light gray for others
            node_colors.append('#00CC66' if on_path else '#AAAAAA')
            n_samples = tree_model.n_node_samples[node_id]
            node_texts.append(f'Leaf\n(n={n_samples})')
            hover_texts.append(f'Leaf Node<br>Samples: {n_samples}<br>Path Length: {depth}')
        else:
            # Decision node
            feature_idx = tree_model.feature[node_id]
            threshold = tree_model.threshold[node_id]
            
            if feature_idx < len(feature_names):
                feat_name = feature_names[feature_idx]
            else:
                feat_name = f'Feature_{feature_idx}'
            
            # Get sample's value for this feature
            sample_value = sample[0, feature_idx] if feature_idx < sample.shape[1] else 0
            goes_left = sample_value <= threshold
            
            if on_path:
                node_colors.append('#CC0000')  # Bright red for path nodes
                direction = "≤" if goes_left else ">"
                node_texts.append(f'{feat_name[:10]}\n{direction} {threshold:.2f}')
            else:
                node_colors.append('#0066CC')  # Bright blue for other nodes
                node_texts.append(f'{feat_name[:10]}\n≤ {threshold:.2f}')
            
            hover_texts.append(
                f'Feature: {feat_name}<br>'
                f'Threshold: {threshold:.3f}<br>'
                f'Sample Value: {sample_value:.3f}<br>'
                f'Goes: {"Left" if goes_left else "Right"}'
            )
        
        # Add edges to children
        left_child = tree_model.children_left[node_id]
        right_child = tree_model.children_right[node_id]
        
        if left_child != -1 and depth < max_depth:
            left_pos = node_positions.get(left_child, {'x': x-0.1, 'level': depth+1})
            x_edges.extend([x, left_pos['x'], None])
            y_edges.extend([y, -left_pos['level'], None])
            add_node(left_child, depth + 1)
        
        if right_child != -1 and depth < max_depth:
            right_pos = node_positions.get(right_child, {'x': x+0.1, 'level': depth+1})
            x_edges.extend([x, right_pos['x'], None])
            y_edges.extend([y, -right_pos['level'], None])
            add_node(right_child, depth + 1)
    
    add_node(0)
    
    # Add edges trace (no row/col needed for single figure)
    fig.add_trace(
        go.Scatter(
            x=x_edges, y=y_edges,
            mode='lines',
            line=dict(color='#7f8c8d', width=3),
            hoverinfo='none',
            showlegend=False,
            name='Edges'
        )
    )
    
    # Add nodes trace (no row/col needed for single figure)
    fig.add_trace(
        go.Scatter(
            x=x_nodes, y=y_nodes,
            mode='markers+text',
            marker=dict(size=200, color=node_colors, line=dict(width=5, color='#333333')),
            text=node_texts,
            textposition='middle center',
            textfont=dict(size=16, color='white', family='Arial Black'),
            hovertext=hover_texts,
            hoverinfo='text',
            showlegend=False,
            name='Nodes'
        )
    )
    
    # Update layout for single tree visualization
    status = "🔴 ANOMALY" if is_anomaly else "🟢 NORMAL"
    fig.update_layout(
        title=dict(
            text=f'Isolation Forest Decision Path - Tree #{best_tree_idx + 1} (Shortest Path)<br>'
                 f'<sup>Score: {sample_score:.4f} | Status: {status} | Path Length: {shortest_path} steps</sup>',
            x=0.5,
            font=dict(size=20)
        ),
        height=1600,
        width=2400,
        paper_bgcolor='#F5F5F5',
        plot_bgcolor='#F5F5F5',
        template='plotly_white',
        showlegend=False
    )
    
    # Hide axes for cleaner tree visualization
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, visible=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, visible=False)
    
    fig.show()
    
    # Print explanation
    print("\n" + "=" * 60)
    print("  ISOLATION FOREST DECISION EXPLANATION")
    print("=" * 60)
    print(f"  Sample Index: {sample_idx}")
    print(f"  Anomaly Score: {sample_score:.4f}")
    print(f"  Prediction: {'ANOMALY' if is_anomaly else 'NORMAL'}")
    print("-" * 60)
    print(f"  Tree Selected: #{best_tree_idx + 1} of {len(all_trees)}")
    print(f"  Path Length: {shortest_path} steps (avg: {avg_path:.1f})")
    print("-" * 60)
    print("  How Isolation Forest Works:")
    print("  • Each tree randomly partitions the feature space")
    print("  • Anomalies are isolated with FEWER splits (shorter paths)")
    print("  • Normal points require MORE splits (longer paths)")
    print("  • Red nodes show the path taken by this sample")
    print("  • This tree isolated the sample fastest → most 'suspicious'")
    print("=" * 60)
    
    return fig


def explain_commit_anomaly(df, commit_hash, model=None, feature_names=None, 
                           X_scaled=None):
    """
    Explain why a specific commit was flagged (or not) as an anomaly.
    
    This provides a detailed breakdown of the Isolation Forest decision
    for a specific commit, showing which features contributed most.
    
    Args:
        df: DataFrame with commit data (must have anomaly detection run)
        commit_hash: Hash of the commit to explain
        model: Trained IsolationForest model (uses _last_model_info if None)
        feature_names: List of feature names (uses _last_model_info if None)
        X_scaled: Scaled feature matrix (uses _last_model_info if None)
        
    Returns:
        Dictionary with explanation details and plotly figure
    """
    global _last_model_info
    
    # Find the commit
    commit_mask = df['hash'].str.startswith(commit_hash[:10])
    if not commit_mask.any():
        print(f"❌ Commit {commit_hash} not found")
        return None
    
    commit_idx = commit_mask.values.argmax()
    commit_row = df.iloc[commit_idx]
    
    # Get model info
    if model is None:
        if not _last_model_info:
            print("❌ No model info available. Run detect_statistical_anomalies first.")
            return None
        model = _last_model_info['model']
        feature_names = _last_model_info['features']
        X_scaled = _last_model_info['X_scaled']
    
    print("=" * 70)
    print(f"  ANOMALY EXPLANATION: {commit_hash[:12]}")
    print("=" * 70)
    print(f"  Author: {commit_row['author']}")
    print(f"  Message: {commit_row['msg_content'][:60]}...")
    print(f"  Date: {commit_row['author_date']}")
    print("-" * 70)
    
    # Show feature values for this commit
    print("\n  Feature Values:")
    for feat in feature_names[:15]:  # Show top 15 features
        if feat in df.columns:
            value = commit_row[feat]
            # Compare to dataset statistics
            mean_val = df[feat].mean()
            std_val = df[feat].std()
            z_score = (value - mean_val) / std_val if std_val > 0 else 0
            
            deviation = ""
            if abs(z_score) > 2:
                deviation = " ⚠️ (unusual)"
            elif abs(z_score) > 1:
                deviation = " ⬆️" if z_score > 0 else " ⬇️"
            
            print(f"    {feat:25s}: {value:10.2f} (z={z_score:+.2f}){deviation}")
    
    # Visualize decision paths
    print("\n  Generating decision path visualization...")
    fig = visualize_isolation_forest_decision_path(
        model, X_scaled, feature_names, commit_idx
    )
    
    return {
        'commit_hash': commit_hash,
        'commit_data': commit_row.to_dict(),
        'figure': fig,
        'is_anomaly': commit_row.get('is_anomaly', None)
    }


# ==========================================
# 3. ML & ANALYSIS FUNCTIONS
# ==========================================

# Store the last trained model info for visualization
_last_model_info = {}

def detect_statistical_anomalies(df, contamination=0.05, use_security_features=True):
    """
    Detect anomalous commits using Isolation Forest.
    
    Args:
        df: DataFrame with commit features
        contamination: Expected proportion of anomalies (default 5%)
        use_security_features: Whether to include security-focused features (default True)
        
    Returns:
        DataFrame with anomaly scores and flags added
    """
    global _last_model_info
    
    if len(df) < 50:
        contamination = 0.1 

    # Base features (always used)
    base_features = [
        'hour_of_day', 
        'lines_inserted', 
        'lines_deleted', 
        'msg_length', 
        'msg_entropy', 
        'churn_ratio', 
        'merge_latency_sec',
        'is_holiday',
        'test_ratio', 
        'sensitive_ratio',
        'files_changed'
    ]
    
    # Security-focused features (improve vulnerability detection)
    security_features = [
        'is_weekend',
        'is_late_night',
        'security_keyword_count',
        'has_fix_keyword',
        'has_security_keyword',
        'has_memory_keyword',
        'security_file_ratio',
        'c_cpp_file_ratio',
        'config_file_ratio',
        'security_file_count',
        'c_cpp_file_count',
    ]
    
    # Select features based on mode
    if use_security_features:
        features = base_features + security_features
        print(f"Using {len(features)} features (base + security)")
    else:
        features = base_features
        print(f"Using {len(features)} base features only")
    
    # Filter to features that exist in the dataframe
    available_features = [f for f in features if f in df.columns]
    if len(available_features) < len(features):
        missing = set(features) - set(available_features)
        print(f"Warning: Missing features: {missing}")
    features = available_features
    
    # Fill NAs
    X = df[features].fillna(0)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(contamination=contamination, random_state=42)
    df['anomaly_score'] = model.fit_predict(X_scaled)
    
    # Get the raw decision scores (lower = more anomalous)
    df['isolation_score'] = model.decision_function(X_scaled)
    
    df['is_anomaly'] = df['anomaly_score'].apply(lambda x: True if x == -1 else False)
    
    # Store model info for visualization
    _last_model_info = {
        'model': model,
        'scaler': scaler,
        'features': features,
        'X_scaled': X_scaled,
        'threshold': model.offset_  # Decision threshold
    }
    
    return df


def detect_dbscan_anomalies(df, eps='auto', min_samples=5, use_security_features=True):
    """
    Detect anomalous commits using DBSCAN clustering.
    
    DBSCAN identifies anomalies as points that don't belong to any cluster (noise points).
    Unlike Isolation Forest, DBSCAN is a density-based method that finds anomalies
    as points in low-density regions.
    
    Args:
        df: DataFrame with commit features
        eps: Maximum distance between samples for neighborhood. 'auto' calculates optimal value.
        min_samples: Minimum samples in neighborhood to form a cluster (default 5)
        use_security_features: Whether to include security-focused features (default True)
        
    Returns:
        DataFrame with anomaly scores and flags added
    """
    global _last_dbscan_info
    
    # Base features (always used)
    base_features = [
        'hour_of_day', 
        'lines_inserted', 
        'lines_deleted', 
        'msg_length', 
        'msg_entropy', 
        'churn_ratio', 
        'merge_latency_sec',
        'is_holiday',
        'test_ratio', 
        'sensitive_ratio',
        'files_changed'
    ]
    
    # Security-focused features (improve vulnerability detection)
    security_features = [
        'is_weekend',
        'is_late_night',
        'security_keyword_count',
        'has_fix_keyword',
        'has_security_keyword',
        'has_memory_keyword',
        'security_file_ratio',
        'c_cpp_file_ratio',
        'config_file_ratio',
        'security_file_count',
        'c_cpp_file_count',
    ]
    
    # Select features based on mode
    if use_security_features:
        features = base_features + security_features
        print(f"DBSCAN: Using {len(features)} features (base + security)")
    else:
        features = base_features
        print(f"DBSCAN: Using {len(features)} base features only")
    
    # Filter to features that exist in the dataframe
    available_features = [f for f in features if f in df.columns]
    if len(available_features) < len(features):
        missing = set(features) - set(available_features)
        print(f"Warning: Missing features: {missing}")
    features = available_features
    
    # Fill NAs
    X = df[features].fillna(0)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Auto-calculate eps using k-distance graph method
    k = min_samples
    nbrs = NearestNeighbors(n_neighbors=k).fit(X_scaled)
    distances, _ = nbrs.kneighbors(X_scaled)
    k_distances = np.sort(distances[:, k-1])  # k-th nearest neighbor distance
    
    if eps == 'auto':
        # Find the "elbow" point - where the distance starts increasing rapidly
        gradient = np.gradient(k_distances)
        gradient2 = np.gradient(gradient)
        elbow_idx = np.argmax(gradient2) if len(gradient2) > 0 else len(k_distances) // 2
        
        # Use a smaller multiplier to make eps more restrictive -> more outliers
        eps = k_distances[elbow_idx] * 0.8
        print(f"DBSCAN: Auto-calculated eps = {eps:.4f}")
    
    # Run DBSCAN
    model = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean', n_jobs=-1)
    cluster_labels = model.fit_predict(X_scaled)
    
    df['dbscan_cluster'] = cluster_labels
    
    # Calculate a "distance score" for ranking
    # Points further from clusters are more anomalous
    unique_clusters = set(cluster_labels) - {-1}  # Exclude noise
    
    if len(unique_clusters) > 0:
        # Calculate centroids of each cluster
        centroids = {}
        for cluster_id in unique_clusters:
            cluster_mask = cluster_labels == cluster_id
            centroids[cluster_id] = np.mean(X_scaled[cluster_mask], axis=0)
        
        # For each point, calculate distance-based score
        dbscan_scores = []
        for i in range(len(X_scaled)):
            if cluster_labels[i] == -1:
                # Noise points: distance to nearest centroid + boost
                min_dist = min(np.linalg.norm(X_scaled[i] - c) for c in centroids.values())
                dbscan_scores.append(min_dist + np.max(k_distances))
            else:
                # Clustered points: distance to own centroid
                own_centroid = centroids[cluster_labels[i]]
                dist = np.linalg.norm(X_scaled[i] - own_centroid)
                dbscan_scores.append(dist)
        
        df['dbscan_score'] = dbscan_scores
    else:
        # All points are noise - use k-distance as score
        df['dbscan_score'] = distances[:, k-1]
    
    # Determine anomalies using score-based threshold (top contamination%)
    # Default: top 15% most anomalous points (matching typical IF contamination)
    contamination = 0.15
    score_threshold = np.percentile(df['dbscan_score'], 100 * (1 - contamination))
    df['is_dbscan_anomaly'] = df['dbscan_score'] >= score_threshold
    
    # Store model info for visualization
    global _last_dbscan_info
    _last_dbscan_info = {
        'model': model,
        'scaler': scaler,
        'features': features,
        'X_scaled': X_scaled,
        'eps': eps,
        'min_samples': min_samples,
        'n_clusters': len(unique_clusters),
        'n_noise': sum(cluster_labels == -1)
    }
    
    n_anomalies = sum(df['is_dbscan_anomaly'])
    print(f"DBSCAN: Found {len(unique_clusters)} clusters and {n_anomalies} noise points (anomalies)")
    print(f"DBSCAN: Anomaly rate = {n_anomalies/len(df)*100:.1f}%")
    
    return df


# Store DBSCAN model info for visualization
_last_dbscan_info = {}


def compare_anomaly_detection_methods(df, contamination=0.05, eps='auto', min_samples=5, 
                                       use_security_features=True, visualize=True):
    """
    Compare Isolation Forest and DBSCAN anomaly detection methods.
    
    This function runs both methods on the same data and provides a comprehensive
    comparison of their results, including overlapping detections and unique findings.
    
    Args:
        df: DataFrame with commit features (output of parse_git_log_file)
        contamination: Expected proportion of anomalies for Isolation Forest (default 5%)
        eps: DBSCAN eps parameter ('auto' for automatic calculation)
        min_samples: DBSCAN min_samples parameter (default 5)
        use_security_features: Whether to include security-focused features (default True)
        visualize: Whether to generate comparison visualizations (default True)
        
    Returns:
        Dictionary with comparison results:
        - 'isolation_forest': DF with IF results
        - 'dbscan': DF with DBSCAN results
        - 'comparison': Summary statistics
        - 'figure': Plotly figure (if visualize=True)
    """
    print("=" * 80)
    print("  ANOMALY DETECTION METHOD COMPARISON")
    print("=" * 80)
    print(f"  Dataset: {len(df)} commits")
    print(f"  Isolation Forest contamination: {contamination*100:.1f}%")
    print(f"  DBSCAN eps: {eps}, min_samples: {min_samples}")
    print("-" * 80)
    
    # Run Isolation Forest
    print("\n📊 Running Isolation Forest...")
    df_if = detect_statistical_anomalies(df.copy(), contamination=contamination, 
                                          use_security_features=use_security_features)
    
    # Run DBSCAN
    print("\n📊 Running DBSCAN...")
    df_dbscan = detect_dbscan_anomalies(df.copy(), eps=eps, min_samples=min_samples,
                                         use_security_features=use_security_features)
    
    # Merge results
    df_combined = df.copy()
    df_combined['is_anomaly_if'] = df_if['is_anomaly']
    df_combined['isolation_score'] = df_if['isolation_score']
    df_combined['is_anomaly_dbscan'] = df_dbscan['is_dbscan_anomaly']
    df_combined['dbscan_score'] = df_dbscan['dbscan_score']
    df_combined['dbscan_cluster'] = df_dbscan['dbscan_cluster']
    
    # Calculate agreement
    both_anomaly = (df_combined['is_anomaly_if'] & df_combined['is_anomaly_dbscan']).sum()
    only_if = (df_combined['is_anomaly_if'] & ~df_combined['is_anomaly_dbscan']).sum()
    only_dbscan = (~df_combined['is_anomaly_if'] & df_combined['is_anomaly_dbscan']).sum()
    both_normal = (~df_combined['is_anomaly_if'] & ~df_combined['is_anomaly_dbscan']).sum()
    
    if_total = df_combined['is_anomaly_if'].sum()
    dbscan_total = df_combined['is_anomaly_dbscan'].sum()
    
    # Agreement metrics
    total = len(df_combined)
    agreement_rate = (both_anomaly + both_normal) / total
    
    # Jaccard similarity for anomaly sets
    union = if_total + dbscan_total - both_anomaly
    jaccard = both_anomaly / union if union > 0 else 0
    
    comparison = {
        'isolation_forest_anomalies': if_total,
        'dbscan_anomalies': dbscan_total,
        'both_methods_agree_anomaly': both_anomaly,
        'only_isolation_forest': only_if,
        'only_dbscan': only_dbscan,
        'both_methods_agree_normal': both_normal,
        'agreement_rate': agreement_rate,
        'jaccard_similarity': jaccard,
        'dbscan_clusters': _last_dbscan_info.get('n_clusters', 0),
        'dbscan_eps': _last_dbscan_info.get('eps', eps),
    }
    
    # Print comparison summary
    print("\n" + "=" * 80)
    print("  COMPARISON RESULTS")
    print("=" * 80)
    print(f"\n  Method Performance:")
    print(f"    Isolation Forest: {if_total} anomalies ({if_total/total*100:.1f}%)")
    print(f"    DBSCAN:           {dbscan_total} anomalies ({dbscan_total/total*100:.1f}%)")
    
    print(f"\n  Agreement Analysis:")
    print(f"    Both detect as anomaly:  {both_anomaly:4d} commits")
    print(f"    Only Isolation Forest:   {only_if:4d} commits")
    print(f"    Only DBSCAN:             {only_dbscan:4d} commits")
    print(f"    Both detect as normal:   {both_normal:4d} commits")
    
    print(f"\n  Agreement Metrics:")
    print(f"    Overall Agreement Rate:  {agreement_rate*100:.1f}%")
    print(f"    Jaccard Similarity:      {jaccard:.3f}")
    print(f"    DBSCAN Clusters Found:   {_last_dbscan_info.get('n_clusters', 0)}")
    
    # Identify high-confidence anomalies (detected by both methods)
    high_confidence = df_combined[
        df_combined['is_anomaly_if'] & df_combined['is_anomaly_dbscan']
    ].copy()
    
    if len(high_confidence) > 0:
        print(f"\n  🔴 High-Confidence Anomalies (detected by both methods):")
        print(f"  {'-' * 60}")
        for _, row in high_confidence.head(10).iterrows():
            print(f"    {row['hash'][:8]} | {row['author'][:15]:<15} | {row['msg_content'][:40]}")
        if len(high_confidence) > 10:
            print(f"    ... and {len(high_confidence) - 10} more")
    
    print("=" * 80)
    
    # Visualization
    fig = None
    if visualize and PLOTLY_AVAILABLE:
        fig = plot_method_comparison(df_combined, comparison)
    
    return {
        'combined_df': df_combined,
        'isolation_forest_df': df_if,
        'dbscan_df': df_dbscan,
        'comparison': comparison,
        'figure': fig
    }


def plot_method_comparison(df, comparison):
    """
    Create visualization comparing Isolation Forest and DBSCAN results.
    
    Args:
        df: Combined DataFrame with results from both methods
        comparison: Dictionary with comparison statistics
        
    Returns:
        Plotly figure
    """
    if not PLOTLY_AVAILABLE:
        print("⚠️ Plotly not available for visualization")
        return None
    
    # Create subplot figure
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Anomaly Detection Agreement (Venn-style)',
            'Score Distributions by Method',
            'Anomalies in Feature Space (PCA)',
            'Detection Overlap'
        ),
        specs=[
            [{"type": "pie"}, {"type": "histogram"}],
            [{"type": "scatter"}, {"type": "bar"}]
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # ============================================
    # Plot 1: Agreement Pie Chart
    # ============================================
    labels = ['Both Anomaly', 'Only IF', 'Only DBSCAN', 'Both Normal']
    values = [
        comparison['both_methods_agree_anomaly'],
        comparison['only_isolation_forest'],
        comparison['only_dbscan'],
        comparison['both_methods_agree_normal']
    ]
    colors = ['#e74c3c', '#3498db', '#9b59b6', '#2ecc71']
    
    fig.add_trace(
        go.Pie(labels=labels, values=values, marker_colors=colors,
               textinfo='label+percent', textposition='inside',
               hole=0.3, showlegend=False),
        row=1, col=1
    )
    
    # ============================================
    # Plot 2: Score Distributions
    # ============================================
    # Normalize scores for comparison (both to 0-1 range, inverted so higher = more anomalous)
    if_scores = df['isolation_score'].values
    if_normalized = (if_scores - if_scores.min()) / (if_scores.max() - if_scores.min() + 1e-10)
    if_normalized = 1 - if_normalized  # Invert so higher = more anomalous
    
    db_scores = df['dbscan_score'].values
    db_normalized = (db_scores - db_scores.min()) / (db_scores.max() - db_scores.min() + 1e-10)
    
    fig.add_trace(
        go.Histogram(x=if_normalized, name='Isolation Forest', 
                    marker_color='#3498db', opacity=0.6, nbinsx=40),
        row=1, col=2
    )
    fig.add_trace(
        go.Histogram(x=db_normalized, name='DBSCAN', 
                    marker_color='#9b59b6', opacity=0.6, nbinsx=40),
        row=1, col=2
    )
    
    # ============================================
    # Plot 3: PCA Scatter with method comparison
    # ============================================
    # Get features used
    if _last_model_info and 'X_scaled' in _last_model_info:
        X_scaled = _last_model_info['X_scaled']
        
        # PCA for visualization
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        
        # Create color categories
        colors_scatter = []
        for i in range(len(df)):
            if df['is_anomaly_if'].iloc[i] and df['is_anomaly_dbscan'].iloc[i]:
                colors_scatter.append('Both')
            elif df['is_anomaly_if'].iloc[i]:
                colors_scatter.append('Only IF')
            elif df['is_anomaly_dbscan'].iloc[i]:
                colors_scatter.append('Only DBSCAN')
            else:
                colors_scatter.append('Normal')
        
        color_map = {
            'Normal': '#2ecc71',
            'Only IF': '#3498db',
            'Only DBSCAN': '#9b59b6',
            'Both': '#e74c3c'
        }
        
        for category in ['Normal', 'Only IF', 'Only DBSCAN', 'Both']:
            mask = [c == category for c in colors_scatter]
            if sum(mask) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=X_pca[mask, 0], y=X_pca[mask, 1],
                        mode='markers',
                        name=category,
                        marker=dict(
                            color=color_map[category],
                            size=8 if category != 'Normal' else 5,
                            opacity=0.8 if category != 'Normal' else 0.4
                        ),
                        showlegend=True
                    ),
                    row=2, col=1
                )
    
    # ============================================
    # Plot 4: Detection Counts Bar Chart
    # ============================================
    methods = ['Isolation Forest', 'DBSCAN', 'Both Methods', 'Either Method']
    counts = [
        comparison['isolation_forest_anomalies'],
        comparison['dbscan_anomalies'],
        comparison['both_methods_agree_anomaly'],
        comparison['isolation_forest_anomalies'] + comparison['dbscan_anomalies'] - comparison['both_methods_agree_anomaly']
    ]
    bar_colors = ['#3498db', '#9b59b6', '#e74c3c', '#f39c12']
    
    fig.add_trace(
        go.Bar(x=methods, y=counts, marker_color=bar_colors, showlegend=False,
               text=counts, textposition='outside'),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title_text='Isolation Forest vs DBSCAN: Anomaly Detection Comparison',
        title_font_size=18,
        height=900,
        width=1200,
        barmode='overlay',
        template='plotly_white',
        showlegend=True
    )
    
    fig.update_xaxes(title_text='Normalized Anomaly Score', row=1, col=2)
    fig.update_yaxes(title_text='Count', row=1, col=2)
    fig.update_xaxes(title_text='PC1', row=2, col=1)
    fig.update_yaxes(title_text='PC2', row=2, col=1)
    fig.update_xaxes(title_text='Detection Method', row=2, col=2)
    fig.update_yaxes(title_text='Anomalies Detected', row=2, col=2)
    
    fig.show()
    return fig


def plot_isolation_forest_explanation(df, highlight_commit=None):
    """
    Visualize how Isolation Forest made its anomaly decisions.
    Uses Plotly for interactive browser-based visualization.
    
    Creates multiple plots:
    1. Anomaly score distribution with decision threshold
    2. Feature importance based on isolation depth
    3. 2D projection showing decision regions (using top 2 features)
    4. Top anomalous commits ranking
    
    Args:
        df: DataFrame with anomaly detection results (must have 'isolation_score' column)
        highlight_commit: Optional commit hash to highlight in visualizations
    """
    global _last_model_info
    
    if not PLOTLY_AVAILABLE:
        print("⚠️ Plotly not available. Install with: pip install plotly")
        return None
    
    if not _last_model_info:
        raise ValueError("No model info available. Run detect_statistical_anomalies first.")
    
    if 'isolation_score' not in df.columns:
        raise ValueError("DataFrame must have 'isolation_score' column. Run detect_statistical_anomalies first.")
    
    model = _last_model_info['model']
    features = _last_model_info['features']
    X_scaled = _last_model_info['X_scaled']
    threshold = _last_model_info['threshold']
    
    # Create 2x2 subplot figure
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Anomaly Score Distribution',
            'Feature Importance for Anomaly Detection',
            'Decision Boundary (Top 2 Features)',
            'Top 15 Most Anomalous Commits'
        ),
        specs=[
            [{"type": "histogram"}, {"type": "bar"}],
            [{"type": "scatter"}, {"type": "bar"}]
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # ============================================
    # Plot 1: Anomaly Score Distribution
    # ============================================
    normal_scores = df[df['is_anomaly'] == False]['isolation_score']
    anomaly_scores = df[df['is_anomaly'] == True]['isolation_score']
    
    fig.add_trace(
        go.Histogram(x=normal_scores, name='Normal', marker_color='#2ecc71', 
                    opacity=0.6, nbinsx=50),
        row=1, col=1
    )
    fig.add_trace(
        go.Histogram(x=anomaly_scores, name='Anomaly', marker_color='#e74c3c', 
                    opacity=0.6, nbinsx=30),
        row=1, col=1
    )
    
    # Decision threshold line
    fig.add_vline(x=threshold, line_dash="dash", line_color="#2c3e50", line_width=2,
                  annotation_text=f"Threshold ({threshold:.3f})", row=1, col=1)
    
    # Highlight specific commit
    if highlight_commit:
        commit_row = df[df['hash'].str.startswith(highlight_commit)]
        if not commit_row.empty:
            commit_score = commit_row.iloc[0]['isolation_score']
            fig.add_vline(x=commit_score, line_color="#9b59b6", line_width=3,
                         annotation_text=f"Commit {highlight_commit[:7]}", row=1, col=1)
    
    # ============================================
    # Plot 2: Feature Importance
    # ============================================
    feature_importance = []
    base_scores = model.decision_function(X_scaled)
    
    for i, feat in enumerate(features):
        X_permuted = X_scaled.copy()
        np.random.seed(42)
        X_permuted[:, i] = np.random.permutation(X_permuted[:, i])
        permuted_scores = model.decision_function(X_permuted)
        importance = np.mean(np.abs(base_scores - permuted_scores))
        feature_importance.append(importance)
    
    sorted_idx = np.argsort(feature_importance)[::-1]
    sorted_features = [features[i] for i in sorted_idx]
    sorted_importance = [feature_importance[i] for i in sorted_idx]
    
    colors = ['#e74c3c' if imp > np.mean(feature_importance) else '#3498db' 
              for imp in sorted_importance]
    
    fig.add_trace(
        go.Bar(y=sorted_features, x=sorted_importance, orientation='h',
               marker_color=colors, name='Importance', showlegend=False),
        row=1, col=2
    )
    
    # ============================================
    # Plot 3: 2D Decision Boundary
    # ============================================
    top_feat_idx = sorted_idx[:2]
    X_2d = X_scaled[:, top_feat_idx]
    
    # Create mesh grid for decision boundary
    x_min, x_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
    y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                         np.linspace(y_min, y_max, 50))
    
    X_mesh = np.zeros((xx.ravel().shape[0], len(features)))
    X_mesh[:, top_feat_idx[0]] = xx.ravel()
    X_mesh[:, top_feat_idx[1]] = yy.ravel()
    for i in range(len(features)):
        if i not in top_feat_idx:
            X_mesh[:, i] = np.mean(X_scaled[:, i])
    
    Z = model.decision_function(X_mesh)
    Z = Z.reshape(xx.shape)
    
    # Contour plot
    fig.add_trace(
        go.Contour(x=np.linspace(x_min, x_max, 50), y=np.linspace(y_min, y_max, 50),
                   z=Z, colorscale='RdYlGn', opacity=0.6, showscale=True,
                   contours=dict(showlines=True), name='Decision Region', showlegend=False),
        row=2, col=1
    )
    
    # Normal commits
    normal_mask = df['is_anomaly'] == False
    fig.add_trace(
        go.Scatter(x=X_2d[normal_mask, 0], y=X_2d[normal_mask, 1],
                   mode='markers', marker=dict(color='#2ecc71', size=6, opacity=0.5),
                   name='Normal', showlegend=False),
        row=2, col=1
    )
    
    # Anomaly commits
    fig.add_trace(
        go.Scatter(x=X_2d[~normal_mask, 0], y=X_2d[~normal_mask, 1],
                   mode='markers', marker=dict(color='#e74c3c', size=10, opacity=0.8,
                                               line=dict(width=1, color='black')),
                   name='Anomaly', showlegend=False),
        row=2, col=1
    )
    
    # Highlight specific commit
    if highlight_commit:
        commit_idx = df[df['hash'].str.startswith(highlight_commit)].index
        if len(commit_idx) > 0:
            idx = df.index.get_loc(commit_idx[0])
            fig.add_trace(
                go.Scatter(x=[X_2d[idx, 0]], y=[X_2d[idx, 1]],
                           mode='markers', marker=dict(color='#9b59b6', size=20, symbol='star',
                                                       line=dict(width=2, color='black')),
                           name=f'Commit {highlight_commit[:7]}', showlegend=False),
                row=2, col=1
            )
    
    # ============================================
    # Plot 4: Top Anomalies Ranking
    # ============================================
    top_anomalies = df.nsmallest(15, 'isolation_score')[['hash', 'isolation_score', 'msg_content', 'is_anomaly']]
    
    colors_bar = ['#e74c3c' if is_anom else '#f39c12' for is_anom in top_anomalies['is_anomaly']]
    labels = [f"{row['hash'][:7]}: {row['msg_content'][:25]}..." 
              for _, row in top_anomalies.iterrows()]
    
    fig.add_trace(
        go.Bar(y=labels, x=top_anomalies['isolation_score'], orientation='h',
               marker_color=colors_bar, name='Score', showlegend=False),
        row=2, col=2
    )
    
    # Threshold line for anomaly ranking
    fig.add_vline(x=threshold, line_dash="dash", line_color="#2c3e50", line_width=2,
                  row=2, col=2)
    
    # Update layout
    fig.update_layout(
        title_text='Isolation Forest Decision Explanation',
        title_font_size=18,
        height=900,
        width=1400,
        barmode='overlay',
        template='plotly_white',
        showlegend=True
    )
    
    # Update axes labels
    fig.update_xaxes(title_text='Isolation Score', row=1, col=1)
    fig.update_xaxes(title_text='Importance', row=1, col=2)
    fig.update_xaxes(title_text=f'{sorted_features[0]} (scaled)', row=2, col=1)
    fig.update_xaxes(title_text='Isolation Score', row=2, col=2)
    fig.update_yaxes(title_text=f'{sorted_features[1]} (scaled)', row=2, col=1)
    
    fig.show()
    return fig

def detect_semantic_anomalies(df, repo_path):
    if not SEMANTIC_AVAILABLE or df.empty: return df

    suspects = df[df['is_anomaly'] == True].copy()
    if suspects.empty: 
        df['semantic_anomaly'] = False
        return df

    def get_diff(h):
        try:
            cmd = ["git", "-C", repo_path, "show", h, "--pretty=", "--minimal"]
            res = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
            return res.stdout[:2000]
        except: return ""

    tqdm.pandas(desc="Analyzing Semantics")
    suspects['diff_content'] = suspects['hash'].progress_apply(lambda h: get_diff(h))
    
    combined_text = (suspects['msg_content'] + "\n" + suspects['diff_content']).tolist()
    embeddings = embedder.encode(combined_text)
    
    if len(embeddings) < 5:
        suspects['semantic_anomaly'] = True
    else:
        iso = IsolationForest(contamination=0.2, random_state=42)
        suspects['semantic_score'] = iso.fit_predict(embeddings)
        suspects['semantic_anomaly'] = suspects['semantic_score'] == -1
        
    df = df.merge(suspects[['hash', 'semantic_anomaly']], on='hash', how='left')
    df['semantic_anomaly'] = df['semantic_anomaly'].fillna(False)
    return df

# ==========================================
# 4. REQUESTED API FUNCTIONS
# ==========================================

# Default features for 3D visualization
DEFAULT_3D_FEATURES = ('lines_inserted', 'lines_deleted', 'churn_ratio')

def get_repo_anomaly_report(repo_path, n_commits=None, ref=None, commit_hash=None, repo_url=None, 
                            use_security_features=True, contamination=0.05, use_cache=True):
    """
    Generate anomaly report for a repository, optionally checking a specific commit.
    
    Args:
        repo_path: Path to the git repository
        n_commits: Number of commits to analyze (None for all)
        ref: Git reference (branch/tag/commit) to start from (None for auto-detect)
        commit_hash: Optional specific commit hash to check within the context
        repo_url: Optional URL to clone from if repo doesn't exist locally
        use_security_features: Use security-focused features for better vuln detection (default True)
        contamination: Expected proportion of anomalies (default 5%)
        use_cache: Whether to use cached parsed logs if available (default True)
        
    Returns:
        DataFrame with commit features and 'is_anomaly' column.
        If commit_hash is provided, also prints details about that specific commit.
    """
    # If commit_hash is provided, use it as the ref to build context around it
    effective_ref = commit_hash if commit_hash else ref
    n_commits_eff = n_commits if n_commits else 500
    
    # Try to load from cache first
    if use_cache and effective_ref:
        cache_key = f"parsed_log_{n_commits_eff}"
        cached_df = load_from_cache(repo_path, effective_ref, cache_key)
        if cached_df is not None:
            print(f"  📦 Using cached git log ({len(cached_df)} commits)")
            df = cached_df
        else:
            # Check if repo exists before attempting git operations
            if not os.path.exists(repo_path) and not repo_url:
                print(f"  ❌ No cache found and repository not available at {repo_path}")
                return None
            # Parse fresh and cache
            log_file = dump_git_log(repo_path, start_ref=effective_ref, n_commits=n_commits, repo_url=repo_url)
            if not log_file: 
                return None
            df = parse_git_log_file(log_file)
            if not df.empty:
                save_to_cache(df, repo_path, effective_ref, cache_key)
                print(f"  💾 Cached git log ({len(df)} commits)")
    else:
        # Check if repo exists before attempting git operations
        if not os.path.exists(repo_path) and not repo_url:
            print(f"  ❌ Repository not available at {repo_path}")
            return None
        log_file = dump_git_log(repo_path, start_ref=effective_ref, n_commits=n_commits, repo_url=repo_url)
        if not log_file: 
            return None
        df = parse_git_log_file(log_file)
    
    if df.empty: 
        return None

    df = detect_statistical_anomalies(df, contamination=contamination, use_security_features=use_security_features)
    
    # If a specific commit was requested, print its details
    if commit_hash:
        target = df[df['hash'].str.startswith(commit_hash)]
        
        if target.empty:
            print(f"Warning: Commit {commit_hash} not found in the parsed log.")
        else:
            row = target.iloc[0]
            
            print("\n" + "="*60)
            print(f"  ANOMALY CHECK: {row['hash']}")
            print("="*60)
            print(f"Author:   {row['author']} <{row['email']}>")
            print(f"Date:     {row['author_date']}")
            print(f"Message:  {row['msg_content']}")
            print("-" * 30)
            print(f"Stats:    +{row['lines_inserted']} / -{row['lines_deleted']} (Churn: {row['churn_ratio']:.2f})")
            print(f"Time:     Holiday={row['is_holiday']} | Weekend={row.get('is_weekend', 0)} | LateNight={row.get('is_late_night', 0)}")
            print(f"Files:    TestRatio={row['test_ratio']:.2f} | SecurityFiles={row.get('security_file_count', 0)} | C/C++={row.get('c_cpp_file_count', 0)}")
            print(f"Message:  SecurityKW={row.get('security_keyword_count', 0)} | FixKW={row.get('has_fix_keyword', 0)} | MemoryKW={row.get('has_memory_keyword', 0)}")
            print("-" * 30)
            
            if row['is_anomaly']:
                print("🔴 STATUS: ANOMALY DETECTED")
                print("   -> This commit deviates statistically from the previous history.")
            else:
                print("🟢 STATUS: NORMAL")
                print("   -> This commit fits the standard profile of this repo.")
    
    return df


# ==========================================
# 5. VULNERABILITY DATASET UTILITIES
# ==========================================

def load_vulnerability_dataset(dataset_path="tool_assisted_manual_dataset.json"):
    """
    Load the vulnerability dataset containing known CVEs with repository and commit info.
    
    Args:
        dataset_path: Path to the JSON dataset file
        
    Returns:
        List of vulnerability records
    """
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    return data


def get_repo_name_from_url(repo_url):
    """Extract repository name from GitHub URL."""
    # Handle various URL formats
    repo_url = repo_url.rstrip('.git').rstrip('/')
    return repo_url.split('/')[-1]


def find_local_repo_path(repo_url):
    """
    Find local path for a repository, checking multiple possible locations.
    
    Args:
        repo_url: GitHub URL of the repository
        
    Returns:
        Local path if found, None otherwise
    """
    repo_name = get_repo_name_from_url(repo_url)
    
    # Check multiple possible locations
    possible_paths = [
        f"./git_repos/{repo_name}",  # Primary location for this project
        f"./{repo_name}",              # Current directory
        f"../git_repos/{repo_name}",   # Parent's git_repos
        f"../{repo_name}",             # Parent directory
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None


def _run_specific_repo_detection(repo_path, repo_url=None, commit_hash=None,
                                  n_commits=400, contamination=0.15,
                                  use_codebert=True, multi_threshold=True,
                                  disable_cache=False, visualize=False):
    """
    Run anomaly detection on a specific repository.
    
    This is a helper function called when specific_repo_path is provided to
    run_vulnerability_detection.
    
    Args:
        repo_path: Path to the git repository
        repo_url: URL to clone from if repo doesn't exist locally
        commit_hash: Specific commit to analyze (optional, uses HEAD if not provided)
        n_commits: Number of commits to analyze
        contamination: Anomaly threshold
        use_codebert: Whether to use CodeBERT embeddings
        multi_threshold: Whether to evaluate at multiple thresholds
        disable_cache: Whether to skip cache loading
        visualize: Whether to generate visualization plots
        
    Returns:
        Dictionary with detection results
    """
    print("=" * 80)
    print("  SPECIFIC REPOSITORY ANALYSIS MODE")
    print("=" * 80)
    
    # Check if repo exists locally
    if not os.path.exists(repo_path):
        if not repo_url:
            raise ValueError(
                f"Repository not found at '{repo_path}' and no repo_url provided. "
                "Please provide specific_repo_url to clone the repository."
            )
        print(f"📥 Repository not found locally. Cloning from {repo_url}...")
        try:
            subprocess.run(['git', 'clone', repo_url, repo_path], 
                         check=True, capture_output=True)
            print(f"✓ Successfully cloned to {repo_path}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to clone repository: {e.stderr.decode()}")
    else:
        print(f"📂 Using local repository: {repo_path}")
    
    # Determine the commit to start from
    if commit_hash:
        start_ref = commit_hash
        print(f"🎯 Analyzing from commit: {commit_hash}")
    else:
        start_ref = get_default_branch(repo_path)
        print(f"🎯 Analyzing from default branch: {start_ref}")
    
    print(f"⚙️  Settings: contamination={contamination*100:.0f}%, n_commits={n_commits}")
    print(f"   CodeBERT: {'enabled' if use_codebert else 'disabled'}")
    print(f"   Cache: {'disabled (refreshing)' if disable_cache else 'enabled'}\n")
    
    # Get parsed log
    df = get_parsed_log_cached(repo_path, start_ref, n_commits, 
                               repo_url=repo_url, disable_cache=disable_cache)
    
    if df is None or df.empty:
        print("❌ Failed to parse git log")
        return {'method_results': {}, 'cve_results': [], 'summary': {}}
    
    print(f"📊 Parsed {len(df)} commits")
    
    # Find target commit position if specified
    target_pos = None
    if commit_hash:
        target_mask = df['hash'].str.startswith(commit_hash[:10])
        if target_mask.any():
            target_pos = target_mask.values.argmax()
            print(f"✓ Target commit found at position {target_pos}")
        else:
            print(f"⚠️ Target commit {commit_hash} not found in parsed log")
    
    # Load CodeBERT if needed
    tokenizer, model = None, None
    if use_codebert:
        try:
            tokenizer, model = get_code_embedding_model()
            if tokenizer is None:
                print("⚠️ CodeBERT not available, skipping embedding method")
                use_codebert = False
        except:
            print("⚠️ CodeBERT not available, skipping embedding method")
            use_codebert = False
    
    # Initialize results
    results = {
        'base': {'detected': [], 'ranks': [], 'percentiles': [], 'scores': None},
        'security_weighted': {'detected': [], 'ranks': [], 'percentiles': [], 'scores': None},
        'dbscan': {'detected': [], 'ranks': [], 'percentiles': [], 'scores': None},
        'embedding_enhanced': {'detected': [], 'ranks': [], 'percentiles': [], 'scores': None},
        'hybrid': {'detected': [], 'ranks': [], 'percentiles': [], 'scores': None}
    }
    
    # ================================================================
    # METHOD 1: Base Features
    # ================================================================
    print("\n📊 Running Base Features method...")
    avail_base = [f for f in BASE_FEATURES if f in df.columns]
    X_base = df[avail_base].fillna(0).values
    X_base_scaled = StandardScaler().fit_transform(X_base)
    
    model_base = IsolationForest(contamination=contamination, n_estimators=200, random_state=42, n_jobs=-1)
    preds_base = model_base.fit_predict(X_base_scaled)
    scores_base = model_base.decision_function(X_base_scaled)
    
    df['base_score'] = scores_base
    df['base_anomaly'] = preds_base == -1
    results['base']['scores'] = scores_base
    
    if target_pos is not None:
        base_detected = preds_base[target_pos] == -1
        base_rank = (scores_base < scores_base[target_pos]).sum() + 1
        base_percentile = base_rank / len(df)
        results['base']['detected'].append(base_detected)
        results['base']['ranks'].append(base_rank)
        results['base']['percentiles'].append(base_percentile)
        print(f"   Target rank: {base_rank}/{len(df)} ({base_percentile*100:.1f}%) {'✅' if base_detected else '❌'}")
    
    # ================================================================
    # METHOD 2: Security Features (Weighted)
    # ================================================================
    print("\n📊 Running Security Features (Weighted) method...")
    avail_sec = [f for f in SECURITY_FEATURES if f in df.columns]
    X_sec = df[avail_sec].fillna(0).values
    weights = np.array([SECURITY_FEATURE_WEIGHTS.get(f, 1.0) for f in avail_sec])
    X_sec_weighted = X_sec * weights
    X_sec_scaled = StandardScaler().fit_transform(X_sec_weighted)
    
    model_sec = IsolationForest(contamination=contamination, n_estimators=200, random_state=42, n_jobs=-1)
    preds_sec = model_sec.fit_predict(X_sec_scaled)
    scores_sec = model_sec.decision_function(X_sec_scaled)
    
    df['security_score'] = scores_sec
    df['security_anomaly'] = preds_sec == -1
    results['security_weighted']['scores'] = scores_sec
    
    if target_pos is not None:
        sec_detected = preds_sec[target_pos] == -1
        sec_rank = (scores_sec < scores_sec[target_pos]).sum() + 1
        sec_percentile = sec_rank / len(df)
        results['security_weighted']['detected'].append(sec_detected)
        results['security_weighted']['ranks'].append(sec_rank)
        results['security_weighted']['percentiles'].append(sec_percentile)
        print(f"   Target rank: {sec_rank}/{len(df)} ({sec_percentile*100:.1f}%) {'✅' if sec_detected else '❌'}")
    
    # ================================================================
    # METHOD 3: DBSCAN (Density-Based)
    # ================================================================
    print("\n📊 Running DBSCAN (Density-Based) method...")
    
    # Use same security features as Isolation Forest for fair comparison
    X_dbscan = df[avail_sec].fillna(0).values
    X_dbscan_scaled = StandardScaler().fit_transform(X_dbscan)
    
    # Auto-calculate eps using k-distance graph method
    min_samples_dbscan = 5
    k = min_samples_dbscan
    nbrs = NearestNeighbors(n_neighbors=k).fit(X_dbscan_scaled)
    distances, _ = nbrs.kneighbors(X_dbscan_scaled)
    k_distances = np.sort(distances[:, k-1])
    
    # Find optimal eps at the elbow
    gradient = np.gradient(k_distances)
    gradient2 = np.gradient(gradient)
    elbow_idx = np.argmax(gradient2) if len(gradient2) > 0 else len(k_distances) // 2
    eps_auto = k_distances[elbow_idx] * 1.2
    print(f"   Auto-calculated eps: {eps_auto:.4f}")
    
    # Run DBSCAN
    model_dbscan = DBSCAN(eps=eps_auto, min_samples=min_samples_dbscan, metric='euclidean', n_jobs=-1)
    cluster_labels = model_dbscan.fit_predict(X_dbscan_scaled)
    
    df['dbscan_cluster'] = cluster_labels
    df['dbscan_anomaly'] = cluster_labels == -1
    
    # Calculate distance-based scores for ranking
    unique_clusters = set(cluster_labels) - {-1}
    n_clusters = len(unique_clusters)
    n_noise = sum(cluster_labels == -1)
    print(f"   Found {n_clusters} clusters, {n_noise} noise points ({n_noise/len(df)*100:.1f}%)")
    
    if n_clusters > 0:
        centroids = {}
        for cluster_id in unique_clusters:
            cluster_mask = cluster_labels == cluster_id
            centroids[cluster_id] = np.mean(X_dbscan_scaled[cluster_mask], axis=0)
        
        dbscan_scores = []
        for i in range(len(X_dbscan_scaled)):
            min_dist = min(np.linalg.norm(X_dbscan_scaled[i] - c) for c in centroids.values())
            dbscan_scores.append(min_dist)
        scores_dbscan = np.array(dbscan_scores)
    else:
        scores_dbscan = np.zeros(len(df))
    
    df['dbscan_score'] = scores_dbscan
    results['dbscan']['scores'] = scores_dbscan
    
    if target_pos is not None:
        dbscan_detected = cluster_labels[target_pos] == -1
        dbscan_rank = (scores_dbscan > scores_dbscan[target_pos]).sum() + 1  # Higher distance = more anomalous
        dbscan_percentile = dbscan_rank / len(df)
        results['dbscan']['detected'].append(dbscan_detected)
        results['dbscan']['ranks'].append(dbscan_rank)
        results['dbscan']['percentiles'].append(dbscan_percentile)
        print(f"   Target rank: {dbscan_rank}/{len(df)} ({dbscan_percentile*100:.1f}%) {'✅' if dbscan_detected else '❌'}")
    
    # ================================================================
    # METHOD 4: Embedding Enhanced
    # ================================================================
    scores_emb = None
    if use_codebert:
        print("\n📊 Running Embedding Enhanced method...")
        try:
            embeddings = get_embeddings_cached(repo_path, df, start_ref, tokenizer, model, 
                                               disable_cache=disable_cache)
            enhanced_emb = compute_enhanced_embedding_features(embeddings, df, avail_sec)
            
            high_signal_sec = ['has_security_keyword', 'has_memory_keyword', 'security_keyword_count',
                              'security_file_count', 'c_cpp_file_count', 'churn_ratio', 
                              'lines_inserted', 'lines_deleted']
            avail_high_signal = [f for f in high_signal_sec if f in df.columns]
            X_high_signal = df[avail_high_signal].fillna(0).values
            
            X_emb_combined = np.hstack([enhanced_emb, X_high_signal])
            X_emb_scaled = StandardScaler().fit_transform(X_emb_combined)
            
            model_emb = IsolationForest(contamination=contamination, n_estimators=200,
                                       max_samples=min(256, len(df)), random_state=42, n_jobs=-1)
            preds_emb = model_emb.fit_predict(X_emb_scaled)
            scores_emb = model_emb.decision_function(X_emb_scaled)
            
            df['embedding_score'] = scores_emb
            df['embedding_anomaly'] = preds_emb == -1
            results['embedding_enhanced']['scores'] = scores_emb
            
            if target_pos is not None:
                emb_detected = preds_emb[target_pos] == -1
                emb_rank = (scores_emb < scores_emb[target_pos]).sum() + 1
                emb_percentile = emb_rank / len(df)
                results['embedding_enhanced']['detected'].append(emb_detected)
                results['embedding_enhanced']['ranks'].append(emb_rank)
                results['embedding_enhanced']['percentiles'].append(emb_percentile)
                print(f"   Target rank: {emb_rank}/{len(df)} ({emb_percentile*100:.1f}%) {'✅' if emb_detected else '❌'}")
        except Exception as e:
            print(f"   ⚠️ Embedding method failed: {str(e)[:50]}")
    
    # ================================================================
    # METHOD 5: Hybrid Ensemble (now includes DBSCAN)
    # ================================================================
    print("\n📊 Computing Hybrid Ensemble...")
    if target_pos is not None and results['base']['percentiles'] and results['security_weighted']['percentiles']:
        base_pct = results['base']['percentiles'][0]
        sec_pct = results['security_weighted']['percentiles'][0]
        dbscan_pct = results['dbscan']['percentiles'][0] if results['dbscan']['percentiles'] else 1.0
        
        if use_codebert and results['embedding_enhanced']['percentiles']:
            emb_pct = results['embedding_enhanced']['percentiles'][0]
            hybrid_pct = min(base_pct, sec_pct, dbscan_pct, emb_pct)
        else:
            hybrid_pct = min(base_pct, sec_pct, dbscan_pct)
        
        hybrid_rank = int(hybrid_pct * len(df)) + 1
        hybrid_detected = hybrid_pct <= contamination
        
        results['hybrid']['detected'].append(hybrid_detected)
        results['hybrid']['ranks'].append(hybrid_rank)
        results['hybrid']['percentiles'].append(hybrid_pct)
        print(f"   Target rank: {hybrid_rank}/{len(df)} ({hybrid_pct*100:.1f}%) {'✅' if hybrid_detected else '❌'}")
    
    # ================================================================
    # Print Summary
    # ================================================================
    print("\n" + "=" * 80)
    print("  ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"  Repository: {repo_path}")
    print(f"  Commits analyzed: {len(df)}")
    print(f"  Anomalies flagged: {sum(df['base_anomaly'])} (IF base), {sum(df['security_anomaly'])} (IF security), {sum(df['dbscan_anomaly'])} (DBSCAN)")
    
    if target_pos is not None:
        print(f"\n  Target Commit: {commit_hash}")
        print(f"  {'─' * 40}")
        if results['base']['ranks']:
            print(f"  Base (IF):      rank {results['base']['ranks'][0]:3d} ({results['base']['percentiles'][0]*100:5.1f}%)")
        if results['security_weighted']['ranks']:
            print(f"  Security (IF):  rank {results['security_weighted']['ranks'][0]:3d} ({results['security_weighted']['percentiles'][0]*100:5.1f}%)")
        if results['dbscan']['ranks']:
            print(f"  DBSCAN:         rank {results['dbscan']['ranks'][0]:3d} ({results['dbscan']['percentiles'][0]*100:5.1f}%)")
        if use_codebert and results['embedding_enhanced']['ranks']:
            print(f"  Embedding(E):   rank {results['embedding_enhanced']['ranks'][0]:3d} ({results['embedding_enhanced']['percentiles'][0]*100:5.1f}%)")
        if results['hybrid']['ranks']:
            print(f"  Hybrid:         rank {results['hybrid']['ranks'][0]:3d} ({results['hybrid']['percentiles'][0]*100:5.1f}%)")
    
    # Show top anomalies comparison
    print(f"\n  Top 10 Anomalies - Isolation Forest (by security score):")
    print(f"  {'─' * 60}")
    top_if = df.nsmallest(10, 'security_score')[['hash', 'author', 'msg_content', 'security_score']]
    for _, row in top_if.iterrows():
        marker = "🎯" if commit_hash and row['hash'].startswith(commit_hash[:10]) else "  "
        print(f"  {marker} {row['hash'][:8]} | {row['author'][:15]:<15} | {row['msg_content'][:40]}")
    
    print(f"\n  Top 10 Anomalies - DBSCAN (by distance score):")
    print(f"  {'─' * 60}")
    top_dbscan = df.nlargest(10, 'dbscan_score')[['hash', 'author', 'msg_content', 'dbscan_score']]
    for _, row in top_dbscan.iterrows():
        marker = "🎯" if commit_hash and row['hash'].startswith(commit_hash[:10]) else "  "
        print(f"  {marker} {row['hash'][:8]} | {row['author'][:15]:<15} | {row['msg_content'][:40]}")
    
    # Method agreement analysis
    if_anomalies = set(df[df['security_anomaly']]['hash'].tolist())
    dbscan_anomalies = set(df[df['dbscan_anomaly']]['hash'].tolist())
    both = if_anomalies & dbscan_anomalies
    only_if = if_anomalies - dbscan_anomalies
    only_dbscan = dbscan_anomalies - if_anomalies
    
    print(f"\n  Method Comparison:")
    print(f"  {'─' * 40}")
    print(f"  Both methods agree (anomaly): {len(both)}")
    print(f"  Only Isolation Forest:        {len(only_if)}")
    print(f"  Only DBSCAN:                  {len(only_dbscan)}")
    
    print("=" * 80)
    
    # ================================================================
    # Visualizations
    # ================================================================
    if visualize:
        print("\n📊 Generating visualizations...")
        
        # Prepare dataframe for visualization functions
        df['is_anomaly'] = df['security_anomaly']
        df['isolation_score'] = df['security_score']
        
        # Add columns for method comparison visualization
        df['is_anomaly_if'] = df['security_anomaly']
        df['is_anomaly_dbscan'] = df['dbscan_anomaly']
        
        # 1. Feature distributions
        try:
            print("   • Feature distributions...")
            plot_feature_distributions(df)
        except Exception as e:
            print(f"   ⚠️ Feature distribution plot failed: {e}")
        
        # 2. 3D commit visualization
        try:
            print("   • 3D commit visualization...")
            plot_commits_3d(
                df, 
                features=DEFAULT_3D_FEATURES,
                highlight_commit=commit_hash,
                title=f"Commit Anomaly Visualization - {os.path.basename(repo_path)}"
            )
        except Exception as e:
            print(f"   ⚠️ 3D plot failed: {e}")
        
        # 3. Isolation Forest explanation
        try:
            print("   • Isolation Forest explanation...")
            # Need to set up _last_model_info for this visualization
            global _last_model_info
            avail_sec = [f for f in SECURITY_FEATURES if f in df.columns]
            X_sec = df[avail_sec].fillna(0).values
            weights = np.array([SECURITY_FEATURE_WEIGHTS.get(f, 1.0) for f in avail_sec])
            X_sec_weighted = X_sec * weights
            X_sec_scaled = StandardScaler().fit_transform(X_sec_weighted)
            
            model_viz = IsolationForest(contamination=contamination, n_estimators=200, random_state=42, n_jobs=-1)
            model_viz.fit(X_sec_scaled)
            
            _last_model_info = {
                'model': model_viz,
                'scaler': StandardScaler().fit(X_sec_weighted),
                'features': avail_sec,
                'X_scaled': X_sec_scaled,
                'threshold': model_viz.offset_
            }
            
            plot_isolation_forest_explanation(df, highlight_commit=commit_hash)
        except Exception as e:
            print(f"   ⚠️ Isolation Forest explanation plot failed: {e}")
        
        # 4. Isolation Forest vs DBSCAN comparison
        try:
            print("   • IF vs DBSCAN comparison...")
            comparison_stats = {
                'isolation_forest_anomalies': len(if_anomalies),
                'dbscan_anomalies': len(dbscan_anomalies),
                'both_methods_agree_anomaly': len(both),
                'only_isolation_forest': len(only_if),
                'only_dbscan': len(only_dbscan),
                'both_methods_agree_normal': len(df) - len(if_anomalies | dbscan_anomalies),
                'agreement_rate': (len(both) + len(df) - len(if_anomalies | dbscan_anomalies)) / len(df),
                'jaccard_similarity': len(both) / len(if_anomalies | dbscan_anomalies) if len(if_anomalies | dbscan_anomalies) > 0 else 0,
                'dbscan_clusters': n_clusters,
                'dbscan_eps': eps_auto,
            }
            plot_method_comparison(df, comparison_stats)
        except Exception as e:
            print(f"   ⚠️ IF vs DBSCAN comparison plot failed: {e}")
        
        print("   ✔ Visualizations complete")

    return {
        'dataframe': df,
        'method_results': results,
        'repo_path': repo_path,
        'target_commit': commit_hash,
        'summary': {
            'total_commits': len(df),
            'base_anomalies': sum(df['base_anomaly']),
            'security_anomalies': sum(df['security_anomaly']),
            'dbscan_anomalies': sum(df['dbscan_anomaly']),
            'both_methods_agree': len(both),
            'dbscan_clusters': n_clusters,
        }
    }


def compute_if_feature_importance(model, X, feature_names):
    """
    Compute feature importance for Isolation Forest based on how often
    each feature is used for splitting and at what depth.
    
    Features used more frequently and at shallower depths are more important
    for isolating anomalies.
    
    Args:
        model: Fitted IsolationForest model
        X: Scaled feature matrix used for fitting
        feature_names: List of feature names corresponding to columns
        
    Returns:
        Dictionary mapping feature names to importance scores
    """
    n_features = len(feature_names)
    importance = np.zeros(n_features)
    
    # Iterate through all trees in the forest
    for tree in model.estimators_:
        tree_struct = tree.tree_
        
        # For each node, get the feature used for splitting
        for node_id in range(tree_struct.node_count):
            # Only internal nodes have splits (not leaves)
            if tree_struct.children_left[node_id] != tree_struct.children_right[node_id]:
                feature_idx = tree_struct.feature[node_id]
                if 0 <= feature_idx < n_features:
                    # Weight by depth: shallower splits are more important
                    # depth approximation: use node samples as proxy
                    depth_weight = 1.0 / (1.0 + np.log1p(node_id))
                    importance[feature_idx] += depth_weight
    
    # Normalize to sum to 1
    total = importance.sum()
    if total > 0:
        importance = importance / total
    
    return {name: float(imp) for name, imp in zip(feature_names, importance)}


def print_feature_importance(feature_importance, n_cves, top_n=10):
    """
    Print aggregated feature importance across all CVEs.
    
    Args:
        feature_importance: Dict with 'base' and 'security' importance dicts
        n_cves: Number of CVEs analyzed (for averaging)
        top_n: Number of top features to display per method
    """
    print("\n")
    print("=" * 80)
    print("  FEATURE IMPORTANCE ANALYSIS (averaged across all CVEs)")
    print("=" * 80)
    
    for method in ['base', 'security']:
        if not feature_importance[method]:
            continue
            
        # Average importance across CVEs
        avg_importance = {
            feat: imp / n_cves 
            for feat, imp in feature_importance[method].items()
        }
        
        # Sort by importance
        sorted_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
        
        method_name = "Base Features (IF)" if method == 'base' else "Security Features (IF)"
        print(f"\n  📊 {method_name}:")
        print(f"  {'Feature':<30} {'Importance':>12} {'Bar':>20}")
        print("  " + "-" * 65)
        
        for feat, imp in sorted_features[:top_n]:
            bar_len = int(imp * 100)
            bar = "█" * min(bar_len, 20)
            print(f"  {feat:<30} {imp*100:>10.1f}%  {bar}")
    
    print("\n" + "=" * 80)


def run_vulnerability_detection(dataset_path="tool_assisted_manual_dataset.json", 
                                 n_samples=None, n_commits=400, contamination=0.15,
                                 cached_only=False, local_repos_only=True,
                                 use_codebert=True, visualize=False,
                                 multi_threshold=False, disable_cache=False,
                                 specific_repo_path=None, specific_repo_url=None,
                                 specific_commit=None, print_results=True):
    """
    Run anomaly detection on vulnerabilities from the dataset, comparing multiple methods.
    
    This is the main entry point for vulnerability detection analysis. It compares
    detection methods using optimal parameters:
    1. Base Features (IF): Isolation Forest with basic statistical features
    2. Security Features (IF): Isolation Forest with security-weighted features  
    3. DBSCAN: Density-based clustering with auto-tuned eps
    4. Enhanced Embedding: CodeBERT embeddings with KNN-based outlier scoring (optional)
    5. Hybrid Ensemble: Combination of all methods with rank fusion
    
    Args:
        dataset_path: Path to the vulnerability dataset JSON file
        n_samples: Number of vulnerabilities to analyze (None = all available)
        n_commits: Number of commits to use as context window
        contamination: Expected anomaly ratio threshold (default 15%)
        cached_only: If True, only analyze repos with fully cached data (fastest)
        local_repos_only: If True, only use repos already cloned locally
        use_codebert: Whether to include CodeBERT embedding comparison
        visualize: If True, generate detailed plots per CVE
        multi_threshold: If True, evaluate at multiple contamination levels
        disable_cache: If True, skip loading from cache and recompute all data
                      (results will still be saved to cache to refresh it)
        specific_repo_path: Path to a specific git repository to analyze (optional)
                           If provided, only this repo will be analyzed instead of dataset
        specific_repo_url: URL to clone the specific repo from if not found locally
                          Required if specific_repo_path is provided but repo doesn't exist
        specific_commit: Specific commit hash to check in the specific repo (optional)
                        If not provided, will analyze the latest commits
        print_results: If True, automatically print all result tables at the end (default True)
                      Set to False for notebook use where you want to print sections separately
        
    Returns:
        Dictionary with detection results, metrics, and multi-threshold analysis
    """
    print("=" * 80)
    print("  ENHANCED VULNERABILITY DETECTION ANALYSIS")
    print("=" * 80)
    print(f"  Cache stats: {get_cache_stats()}")
    print()
    
    # Handle specific repository mode
    if specific_repo_path is not None:
        return _run_specific_repo_detection(
            repo_path=specific_repo_path,
            repo_url=specific_repo_url,
            commit_hash=specific_commit,
            n_commits=n_commits,
            contamination=contamination,
            use_codebert=use_codebert,
            multi_threshold=multi_threshold,
            disable_cache=disable_cache,
            visualize=visualize
        )
    
    # Load vulnerabilities
    vulnerabilities = load_vulnerability_dataset(dataset_path)
    
    # Filter to valid vulnerabilities with introducing commits
    valid_vulns = [v for v in vulnerabilities if v.get('introducing') and v.get('repository')]
    print(f"📚 Loaded {len(vulnerabilities)} vulnerabilities, {len(valid_vulns)} have valid introducing commits")
    
    # If cached_only, filter to vulns with cached data
    if cached_only:
        samples = []
        for v in valid_vulns:
            repo_name = get_repo_name_from_url(v['repository'])
            commit_hash = v['introducing']
            cache_key = f"parsed_log_{n_commits}"
            cache_repo_path = f"./git_repos/{repo_name}"
            if is_cached(cache_repo_path, commit_hash, cache_key):
                samples.append((v, cache_repo_path))
        print(f"🗃️  {len(samples)} have cached data (cached_only=True)")
        
        if not samples:
            print("❌ No cached vulnerabilities found! Run without cached_only first.")
            return {'method_results': {}, 'cve_results': [], 'summary': {}}
        
        print(f"📋 Using all {len(samples)} cached vulnerabilities\n")
    
    # If local_repos_only, filter to repos that exist locally
    elif local_repos_only:
        samples = []
        for v in valid_vulns:
            repo_path = get_local_repo_path(v['repository'])
            if repo_path:
                samples.append((v, repo_path))
        print(f"📁 Found {len(samples)} vulnerabilities in existing local repos")
        
        if not samples:
            print("❌ No local repos found! Clone repos or run with cached_only=True.")
            return {'method_results': {}, 'cve_results': [], 'summary': {}}
        
        if n_samples and n_samples < len(samples):
            samples = random.sample(samples, n_samples)
        print(f"🎯 Testing on {len(samples)} vulnerabilities\n")
    
    else:
        samples = []
        for v in valid_vulns:
            repo_name = get_repo_name_from_url(v['repository'])
            repo_path = find_local_repo_path(v['repository']) or f"./git_repos/{repo_name}"
            samples.append((v, repo_path))
        
        if n_samples and n_samples < len(samples):
            samples = random.sample(samples, n_samples)
        print(f"🎯 Testing on {len(samples)} vulnerabilities\n")
    
    print(f"⚙️  Settings: contamination={contamination*100:.0f}%, n_commits={n_commits}")
    print(f"   Methods: Isolation Forest + DBSCAN (auto-tuned eps)")
    print(f"   CodeBERT: {'enabled' if use_codebert else 'disabled'}")
    print(f"   Cache: {'disabled (refreshing)' if disable_cache else 'enabled'}\n")
    
    # Load CodeBERT if needed
    tokenizer, model = None, None
    if use_codebert:
        try:
            tokenizer, model = get_code_embedding_model()
            if tokenizer is None:
                print("⚠️ CodeBERT not available, skipping embedding comparison")
                use_codebert = False
        except:
            print("⚠️ CodeBERT not available, skipping embedding comparison")
            use_codebert = False
    
    # Results storage - now includes rankings for multi-threshold evaluation
    results = {
        'base': {'detected': [], 'ranks': [], 'percentiles': []},
        'security_weighted': {'detected': [], 'ranks': [], 'percentiles': []},
        'dbscan': {'detected': [], 'ranks': [], 'percentiles': []},
        'embedding_enhanced': {'detected': [], 'ranks': [], 'percentiles': []},
        'hybrid': {'detected': [], 'ranks': [], 'percentiles': []}
    }
    
    # Feature importance accumulators (sum across all CVEs)
    feature_importance = {
        'base': {},      # feature_name -> cumulative importance
        'security': {},  # feature_name -> cumulative importance
    }
    
    # Multi-threshold results
    multi_thresh_results = {level: {'base': 0, 'security_weighted': 0, 'dbscan': 0, 'embedding_enhanced': 0, 'hybrid': 0} 
                           for level in CONTAMINATION_LEVELS}
    
    # Per-CVE detailed results
    cve_results = []
    
    for i, (vuln, repo_path) in enumerate(samples, 1):
        cve = vuln.get('cve', 'Unknown')
        cwe = vuln.get('cwe', 'Unknown')
        repo_url = vuln['repository']
        commit = vuln['introducing']
        repo_name = get_repo_name_from_url(repo_url)
        
        print("=" * 80)
        print(f"  [{i}/{len(samples)}] {cve}")
        print("=" * 80)
        print(f"  CWE: {cwe} | Repository: {repo_name}")
        print(f"  Commit: {commit}")
        print("-" * 80)
        
        if repo_path is None:
            print("  ⚠️ Repo not available, skipping")
            continue
        
        try:
            # Get parsed log (cached unless disable_cache is True)
            df = get_parsed_log_cached(repo_path, commit, n_commits, repo_url=repo_url, disable_cache=disable_cache)
            
            if df is None or len(df) < 30:
                print(f"  ⚠️ Insufficient commits ({len(df) if df is not None else 0}), skipping")
                continue
            
            # Find target commit - convert iloc index to position for reliable indexing
            target_mask = df['hash'].str.startswith(commit[:10])
            if not target_mask.any():
                print("  ⚠️ Target commit not found in log")
                continue
            
            # Get position-based index
            target_pos = target_mask.values.argmax()
            
            # ================================================================
            # METHOD 1: Base Features (Baseline)
            # ================================================================
            avail_base = [f for f in BASE_FEATURES if f in df.columns]
            X_base = df[avail_base].fillna(0).values
            X_base_scaled = StandardScaler().fit_transform(X_base)
            
            model_base = IsolationForest(contamination=contamination, n_estimators=200, random_state=42, n_jobs=-1)
            preds_base = model_base.fit_predict(X_base_scaled)
            scores_base = model_base.decision_function(X_base_scaled)
            
            # Compute feature importance for Base IF
            # Use mean decrease in path length as proxy for importance
            base_importances = compute_if_feature_importance(model_base, X_base_scaled, avail_base)
            for feat, imp in base_importances.items():
                feature_importance['base'][feat] = feature_importance['base'].get(feat, 0) + imp
            
            base_detected = preds_base[target_pos] == -1
            base_rank = (scores_base < scores_base[target_pos]).sum() + 1
            base_percentile = base_rank / len(df)
            
            results['base']['detected'].append(base_detected)
            results['base']['ranks'].append(base_rank)
            results['base']['percentiles'].append(base_percentile)
            
            # ================================================================
            # METHOD 2: Security Features (Base + security-specific)
            # ================================================================
            avail_sec = [f for f in SECURITY_FEATURES if f in df.columns]
            # Get only the additional security features (not in base)
            extra_sec = [f for f in avail_sec if f not in avail_base]
            
            # Debug: print feature counts for first CVE
            if i == 1:
                print(f"  📋 Features: Base={len(avail_base)}, Security={len(avail_sec)} (+{len(extra_sec)} extra)")
            
            X_sec = df[avail_sec].fillna(0).values
            X_sec_scaled = StandardScaler().fit_transform(X_sec)
            
            model_sec = IsolationForest(contamination=contamination, n_estimators=200, random_state=42, n_jobs=-1)
            preds_sec = model_sec.fit_predict(X_sec_scaled)
            scores_sec = model_sec.decision_function(X_sec_scaled)
            
            # Compute feature importance for Security IF
            sec_importances = compute_if_feature_importance(model_sec, X_sec_scaled, avail_sec)
            for feat, imp in sec_importances.items():
                feature_importance['security'][feat] = feature_importance['security'].get(feat, 0) + imp
            
            sec_detected = preds_sec[target_pos] == -1
            sec_rank = (scores_sec < scores_sec[target_pos]).sum() + 1
            sec_percentile = sec_rank / len(df)
            
            results['security_weighted']['detected'].append(sec_detected)
            results['security_weighted']['ranks'].append(sec_rank)
            results['security_weighted']['percentiles'].append(sec_percentile)
            
            # ================================================================
            # METHOD 3: DBSCAN (Density-Based)
            # ================================================================
            # Use same security features as Isolation Forest for fair comparison
            X_dbscan = df[avail_sec].fillna(0).values
            X_dbscan_scaled = StandardScaler().fit_transform(X_dbscan)
            
            # Auto-calculate eps using k-distance graph method
            min_samples_dbscan = max(3, min(5, len(df) // 50))  # Adaptive min_samples
            k = min_samples_dbscan
            nbrs = NearestNeighbors(n_neighbors=k).fit(X_dbscan_scaled)
            distances, _ = nbrs.kneighbors(X_dbscan_scaled)
            k_distances = np.sort(distances[:, k-1])
            
            # Find optimal eps at the elbow - use a more aggressive eps to produce more outliers
            gradient = np.gradient(k_distances)
            gradient2 = np.gradient(gradient)
            elbow_idx = np.argmax(gradient2) if len(gradient2) > 0 else len(k_distances) // 2
            # Use a smaller multiplier to make eps more restrictive -> more outliers
            eps_auto = k_distances[elbow_idx] * 0.8
            
            # Run DBSCAN
            model_dbscan = DBSCAN(eps=eps_auto, min_samples=min_samples_dbscan, metric='euclidean', n_jobs=-1)
            cluster_labels = model_dbscan.fit_predict(X_dbscan_scaled)
            
            # Calculate distance-based anomaly scores for ranking
            # Points far from cluster centroids or noise points (-1) are more anomalous
            unique_clusters = set(cluster_labels) - {-1}
            n_noise = sum(cluster_labels == -1)
            
            if len(unique_clusters) > 0:
                centroids = {}
                for cluster_id in unique_clusters:
                    cluster_mask = cluster_labels == cluster_id
                    centroids[cluster_id] = np.mean(X_dbscan_scaled[cluster_mask], axis=0)
                
                dbscan_scores = []
                for idx in range(len(X_dbscan_scaled)):
                    if cluster_labels[idx] == -1:
                        # Noise points get high score (anomalous) based on distance to nearest centroid
                        min_dist = min(np.linalg.norm(X_dbscan_scaled[idx] - c) for c in centroids.values())
                        # Boost noise points to ensure they rank higher than cluster points
                        dbscan_scores.append(min_dist + np.max(k_distances))
                    else:
                        # For clustered points, score is distance to own cluster centroid
                        own_centroid = centroids[cluster_labels[idx]]
                        dist = np.linalg.norm(X_dbscan_scaled[idx] - own_centroid)
                        dbscan_scores.append(dist)
                scores_dbscan = np.array(dbscan_scores)
            else:
                # All points are noise - use k-distance as score
                scores_dbscan = distances[:, k-1]
            
            # Rank: higher distance = more anomalous = lower rank number
            dbscan_rank = (scores_dbscan > scores_dbscan[target_pos]).sum() + 1
            dbscan_percentile = dbscan_rank / len(df)
            
            # Detection: use score-based threshold to match contamination rate
            # This makes DBSCAN detection comparable to Isolation Forest
            score_threshold = np.percentile(scores_dbscan, 100 * (1 - contamination))
            dbscan_detected = scores_dbscan[target_pos] >= score_threshold
            
            results['dbscan']['detected'].append(dbscan_detected)
            results['dbscan']['ranks'].append(dbscan_rank)
            results['dbscan']['percentiles'].append(dbscan_percentile)
            
            # ================================================================
            # METHOD 4: Enhanced Embedding Features
            # ================================================================
            emb_detected = False
            emb_rank = len(df)
            emb_percentile = 1.0
            scores_emb = None
            
            if use_codebert:
                try:
                    embeddings = get_embeddings_cached(repo_path, df, commit, tokenizer, model, disable_cache=disable_cache)
                    
                    # Get enhanced embedding features (includes security-modulated scores)
                    enhanced_emb = compute_enhanced_embedding_features(embeddings, df, avail_sec)
                    
                    # Combine with high-signal security features (not all security features)
                    high_signal_sec = ['has_security_keyword', 'has_memory_keyword', 'security_keyword_count',
                                      'security_file_count', 'c_cpp_file_count', 'churn_ratio', 
                                      'lines_inserted', 'lines_deleted']
                    avail_high_signal = [f for f in high_signal_sec if f in df.columns]
                    X_high_signal = df[avail_high_signal].fillna(0).values
                    
                    # Combine embedding features with high-signal security features
                    X_emb_combined = np.hstack([enhanced_emb, X_high_signal])
                    X_emb_scaled = StandardScaler().fit_transform(X_emb_combined)
                    
                    model_emb = IsolationForest(contamination=contamination, n_estimators=200, 
                                               max_samples=min(256, len(df)), random_state=42, n_jobs=-1)
                    preds_emb = model_emb.fit_predict(X_emb_scaled)
                    scores_emb = model_emb.decision_function(X_emb_scaled)
                    
                    emb_detected = preds_emb[target_pos] == -1
                    emb_rank = (scores_emb < scores_emb[target_pos]).sum() + 1
                    emb_percentile = emb_rank / len(df)
                except Exception as emb_err:
                    print(f"  ⚠️ Embedding error: {str(emb_err)[:40]}")
            
            results['embedding_enhanced']['detected'].append(emb_detected)
            results['embedding_enhanced']['ranks'].append(emb_rank)
            results['embedding_enhanced']['percentiles'].append(emb_percentile)
            
            # ================================================================
            # METHOD 5: Hybrid Ensemble (Best of IF methods)
            # ================================================================
            # Use min-rank fusion: take the best rank from Base IF and Security IF
            # DBSCAN excluded from hybrid as it uses different detection paradigm
            base_rank_pct = base_percentile
            sec_rank_pct = sec_percentile
            
            if use_codebert and scores_emb is not None:
                emb_rank_pct = emb_percentile
                # Min-rank: best of Base, Security, and Embedding
                min_rank_pct = min(base_rank_pct, sec_rank_pct, emb_rank_pct)
            else:
                # Min-rank: best of Base and Security only
                min_rank_pct = min(base_rank_pct, sec_rank_pct)
            
            # Use min-rank for the hybrid (captures ANY IF method detecting it)
            hybrid_percentile = min_rank_pct
            hybrid_rank = int(hybrid_percentile * len(df)) + 1
            hybrid_detected = hybrid_percentile <= contamination
            
            results['hybrid']['detected'].append(hybrid_detected)
            results['hybrid']['ranks'].append(hybrid_rank)
            results['hybrid']['percentiles'].append(hybrid_percentile)
            
            # ================================================================
            # Multi-threshold evaluation
            # ================================================================
            if multi_threshold:
                for level in CONTAMINATION_LEVELS:
                    n_anomalies = int(len(df) * level)
                    if base_rank <= n_anomalies:
                        multi_thresh_results[level]['base'] += 1
                    if sec_rank <= n_anomalies:
                        multi_thresh_results[level]['security_weighted'] += 1
                    if dbscan_rank <= n_anomalies:
                        multi_thresh_results[level]['dbscan'] += 1
                    if emb_rank <= n_anomalies:
                        multi_thresh_results[level]['embedding_enhanced'] += 1
                    if hybrid_rank <= n_anomalies:
                        multi_thresh_results[level]['hybrid'] += 1
            
            # Store per-CVE result
            cve_result = {
                'cve': cve,
                'cwe': cwe,
                'repository': repo_url,
                'commit': commit,
                'base': base_detected,
                'base_rank': base_rank,
                'base_percentile': base_percentile,
                'security_weighted': sec_detected,
                'security_rank': sec_rank,
                'security_percentile': sec_percentile,
                'dbscan': dbscan_detected,
                'dbscan_rank': dbscan_rank,
                'dbscan_percentile': dbscan_percentile,
                'embedding_enhanced': emb_detected,
                'embedding_rank': emb_rank,
                'embedding_percentile': emb_percentile,
                'hybrid': hybrid_detected,
                'hybrid_rank': hybrid_rank,
                'hybrid_percentile': hybrid_percentile,
                'total_commits': len(df),
            }
            cve_results.append(cve_result)
            
            # Print per-CVE result with rankings
            print(f"  📊 Rankings (lower = more anomalous):")
            print(f"     Base (IF):      rank {base_rank:3d}/{len(df)} ({base_percentile*100:5.1f}%) {'✅' if base_detected else '❌'}")
            print(f"     Security (IF):  rank {sec_rank:3d}/{len(df)} ({sec_percentile*100:5.1f}%) {'✅' if sec_detected else '❌'}")
            print(f"     DBSCAN:         rank {dbscan_rank:3d}/{len(df)} ({dbscan_percentile*100:5.1f}%) {'✅' if dbscan_detected else '❌'}")
            if use_codebert:
                print(f"     Embedding(E):   rank {emb_rank:3d}/{len(df)} ({emb_percentile*100:5.1f}%) {'✅' if emb_detected else '❌'}")
            print(f"     Hybrid:         rank {hybrid_rank:3d}/{len(df)} ({hybrid_percentile*100:5.1f}%) {'✅' if hybrid_detected else '❌'}")
            
            # Print noise points info for DBSCAN
            n_noise = sum(cluster_labels == -1)
            n_clusters = len(unique_clusters)
            print(f"     DBSCAN info:    {n_clusters} clusters, {n_noise} noise pts ({n_noise/len(df)*100:.1f}%)")
            
        except Exception as e:
            import traceback
            print(f"  ❌ Error: {str(e)[:60]}")
            traceback.print_exc()
        
        print()
    
    # ================================================================
    # RESULTS ANALYSIS
    # ================================================================
    
    # Sort CVE results by hybrid percentile (best detection method)
    cve_results_sorted = sorted(cve_results, key=lambda x: x['hybrid_percentile'])
    
    # Store all results for modular printing
    all_results = {
        'method_results': results,
        'cve_results': cve_results_sorted,
        'multi_threshold': multi_thresh_results if multi_threshold else None,
        'feature_importance': feature_importance,
        'contamination': contamination,
        'use_codebert': use_codebert,
        'summary': {
            method: {
                'detected': sum(data['detected']),
                'total': len(data['detected']),
                'mean_percentile': np.mean(data['percentiles']) if data['percentiles'] else 0,
                'median_rank': np.median(data['ranks']) if data['ranks'] else 0
            }
            for method, data in results.items()
        }
    }
    
    # Print all results (can be called separately via print_all_results)
    if print_results:
        print_all_results(all_results)
    
    return all_results


def print_all_results(results):
    """
    Print all benchmark results. Can be called separately after benchmark completes.
    
    Args:
        results: Dictionary returned by run_vulnerability_detection
    """
    cve_results = results['cve_results']
    use_codebert = results['use_codebert']
    contamination = results['contamination']
    multi_thresh_results = results['multi_threshold']
    feature_importance = results['feature_importance']
    method_results = results['method_results']
    
    # Print enhanced comparison table (CVE-wise breakdown)
    print("\n")
    print_enhanced_comparison_table(cve_results, use_codebert, contamination)
    
    # Print multi-threshold analysis
    if multi_thresh_results:
        print_multi_threshold_analysis(multi_thresh_results, len(cve_results), use_codebert)
    
    # Compute and print aggregate metrics (overall comparison)
    print_aggregate_metrics(method_results, contamination, use_codebert)
    
    # Print feature importance analysis
    n_cves = len(cve_results)
    if n_cves > 0:
        print_feature_importance(feature_importance, n_cves)


def print_cve_comparison_table(results):
    """Print only the CVE-wise comparison table."""
    print_enhanced_comparison_table(
        results['cve_results'], 
        results['use_codebert'], 
        results['contamination']
    )


def print_overall_comparison(results):
    """Print only the overall/aggregate comparison metrics."""
    print_aggregate_metrics(
        results['method_results'], 
        results['contamination'], 
        results['use_codebert']
    )


def print_feature_contributions(results):
    """Print only the feature importance/contribution analysis."""
    n_cves = len(results['cve_results'])
    if n_cves > 0:
        print_feature_importance(results['feature_importance'], n_cves)


# ==========================================
# 6. CODE EMBEDDING ENHANCEMENT (PLACEHOLDER)
# ==========================================

# Flag for whether code embedding model is available
CODE_EMBEDDING_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    CODE_EMBEDDING_AVAILABLE = True
except ImportError:
    pass

def get_code_embedding_model():
    """
    Load a pre-trained code embedding model.
    Using Microsoft's CodeBERT or similar for semantic code understanding.
    
    Returns:
        Tuple of (tokenizer, model) or (None, None) if not available
    """
    if not CODE_EMBEDDING_AVAILABLE:
        print("⚠️ Code embedding not available. Install transformers and torch:")
        print("   pip install transformers torch")
        return None, None
    
    try:
        # CodeBERT is trained on code and can understand semantic patterns
        model_name = "microsoft/codebert-base"
        print(f"Loading code embedding model: {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()
        print("✓ Code embedding model loaded")
        return tokenizer, model
    except Exception as e:
        print(f"⚠️ Failed to load code embedding model: {e}")
        return None, None


def get_commit_diff(repo_path, commit_hash, use_cache=True):
    """
    Get the actual code diff for a commit.
    
    Args:
        repo_path: Path to git repository
        commit_hash: Commit hash to get diff for
        use_cache: Whether to use caching (default True)
        
    Returns:
        String containing the diff, or None if failed
    """
    # Note: Individual diffs are small, we cache them as part of batch operations
    try:
        result = subprocess.run(
            ['git', '-C', repo_path, 'show', '--format=', '-p', commit_hash],
            capture_output=True, timeout=30
        )
        if result.returncode == 0:
            # Handle encoding issues gracefully
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    return result.stdout.decode(encoding)
                except:
                    continue
            return result.stdout.decode('utf-8', errors='replace')
    except Exception as e:
        pass  # Silent fail for individual diffs
    return None


def get_all_commit_diffs_cached(repo_path, commit_hashes, start_commit):
    """
    Get diffs for all commits, using cache if available.
    
    Args:
        repo_path: Path to git repository
        commit_hashes: List of commit hashes to get diffs for
        start_commit: Starting commit (used as cache key)
        
    Returns:
        Dictionary mapping commit hash to diff text
    """
    # Try to load from cache
    cached = load_from_cache(repo_path, start_commit, 'diffs')
    if cached is not None:
        print(f"  📦 Loaded {len(cached)} diffs from cache")
        return cached
    
    # Compute diffs
    print(f"  ⏳ Computing diffs for {len(commit_hashes)} commits...")
    diffs = {}
    for h in tqdm(commit_hashes, desc="Getting diffs", leave=False):
        diffs[h] = get_commit_diff(repo_path, h, use_cache=False)
    
    # Save to cache
    save_to_cache(diffs, repo_path, start_commit, 'diffs')
    print(f"  💾 Cached {len(diffs)} diffs")
    
    return diffs


def compute_code_embedding(diff_text, tokenizer, model, max_length=512):
    """
    Compute embedding for a code diff using CodeBERT.
    
    Args:
        diff_text: The diff text to embed
        tokenizer: CodeBERT tokenizer
        model: CodeBERT model
        max_length: Maximum sequence length
        
    Returns:
        numpy array of embeddings (768 dimensions for CodeBERT)
    """
    if tokenizer is None or model is None:
        return None
    
    # Truncate diff to reasonable size
    if len(diff_text) > 10000:
        diff_text = diff_text[:10000]
    
    with torch.no_grad():
        inputs = tokenizer(
            diff_text, 
            return_tensors="pt", 
            max_length=max_length, 
            truncation=True,
            padding=True
        )
        outputs = model(**inputs)
        # Use [CLS] token embedding as sentence/diff representation
        embedding = outputs.last_hidden_state[:, 0, :].numpy()
    
    return embedding.flatten()


def get_embeddings_cached(repo_path, df, start_commit, tokenizer=None, model=None, disable_cache=False):
    """
    Get CodeBERT embeddings for all commits, using cache if available.
    
    Args:
        repo_path: Path to git repository
        df: DataFrame with commit data (must have 'hash' column)
        start_commit: Starting commit (used as cache key)
        tokenizer: CodeBERT tokenizer (loaded if None)
        model: CodeBERT model (loaded if None)
        disable_cache: If True, skip loading from cache (still saves to cache)
        
    Returns:
        numpy array of shape (n_commits, 768) with embeddings
    """
    # Try to load from cache (unless disabled)
    if not disable_cache:
        cached = load_from_cache(repo_path, start_commit, 'embeddings')
        if cached is not None:
            # Verify cache matches current df
            if len(cached) == len(df):
                print(f"  📦 Loaded {len(cached)} embeddings from cache")
                return cached
            else:
                print(f"  ⚠️ Cache size mismatch, recomputing...")
    else:
        print(f"  🔄 Cache disabled, recomputing embeddings...")
    
    # Load model if needed
    if tokenizer is None or model is None:
        tokenizer, model = get_code_embedding_model()
        if tokenizer is None:
            print("  ❌ CodeBERT not available")
            return np.zeros((len(df), 768))
    
    # Get diffs (also cached)
    commit_hashes = df['hash'].tolist()
    diffs = get_all_commit_diffs_cached(repo_path, commit_hashes, start_commit)
    
    # Compute embeddings
    print(f"  ⏳ Computing embeddings for {len(df)} commits...")
    embeddings = []
    for h in tqdm(commit_hashes, desc="Computing embeddings", leave=False):
        diff = diffs.get(h, "")
        if diff:
            emb = compute_code_embedding(diff, tokenizer, model)
            embeddings.append(emb if emb is not None else np.zeros(768))
        else:
            embeddings.append(np.zeros(768))
    
    embedding_matrix = np.array(embeddings)
    
    # Save to cache
    save_to_cache(embedding_matrix, repo_path, start_commit, 'embeddings')
    print(f"  💾 Cached {len(embeddings)} embeddings")
    
    return embedding_matrix


def get_parsed_log_cached(repo_path, start_commit, n_commits=500, repo_url=None, disable_cache=False):
    """
    Get parsed git log, using cache if available.
    
    Args:
        repo_path: Path to git repository
        start_commit: Starting commit hash
        n_commits: Number of commits to parse
        repo_url: URL to clone repository from if not found locally
        disable_cache: If True, skip loading from cache (still saves to cache)
        
    Returns:
        DataFrame with parsed commit data
    """
    cache_key = f"parsed_log_{n_commits}"
    
    # Try to load from cache (unless disabled)
    if not disable_cache:
        cached = load_from_cache(repo_path, start_commit, cache_key)
        
        if cached is not None:
            print(f"  📦 Loaded parsed log ({len(cached)} commits) from cache")
            return cached
    else:
        print(f"  🔄 Cache disabled, recomputing parsed log...")
    
    # Parse fresh (pass repo_url for auto-cloning if needed)
    log_file = dump_git_log(repo_path, start_ref=start_commit, n_commits=n_commits, repo_url=repo_url)
    df = parse_git_log_file(log_file)
    
    if not df.empty:
        # Save to cache
        save_to_cache(df, repo_path, start_commit, cache_key)
        print(f"  💾 Cached parsed log ({len(df)} commits)")
    
    return df


# ==========================================
# 7. ENHANCED DETECTION METHODS
# ==========================================

def extract_vulnerability_features_from_diff(diff_text):
    """
    Extract vulnerability-specific features from code diff.
    
    These features are designed to capture patterns commonly seen in
    vulnerability-introducing commits.
    
    Args:
        diff_text: Raw git diff text
        
    Returns:
        Dictionary of vulnerability-specific features
    """
    if not diff_text:
        return {
            'dangerous_function_count': 0,
            'memory_op_count': 0,
            'bounds_check_removed': 0,
            'error_handling_changed': 0,
            'auth_code_changed': 0,
            'crypto_code_changed': 0,
            'input_handling_changed': 0,
            'format_string_count': 0,
            'pointer_arithmetic': 0,
            'null_check_removed': 0,
            'added_lines_ratio': 0,
            'removed_lines_ratio': 0,
            'complexity_indicator': 0,
        }
    
    diff_lower = diff_text.lower()
    
    # Dangerous C/C++ functions (buffer overflow risks)
    dangerous_funcs = ['strcpy', 'strcat', 'sprintf', 'gets', 'scanf', 
                       'memcpy', 'memmove', 'strncpy', 'strncat',
                       'alloca', 'realpath', 'getwd']
    dangerous_count = sum(diff_lower.count(func) for func in dangerous_funcs)
    
    # Memory operations
    memory_ops = ['malloc', 'calloc', 'realloc', 'free', 'new', 'delete',
                  'alloc', 'dealloc', 'mmap', 'munmap']
    memory_count = sum(diff_lower.count(op) for op in memory_ops)
    
    # Bounds checking patterns (removal is concerning)
    bounds_patterns = ['< len', '< size', '<= len', '<= size', 
                       'bounds', 'limit', 'max_', 'min_', 'range']
    bounds_removed = sum(1 for p in bounds_patterns if f'-{p}' in diff_lower or f'- {p}' in diff_lower)
    
    # Error handling changes
    error_patterns = ['try', 'catch', 'throw', 'except', 'error', 
                      'exception', 'finally', 'errno']
    error_changed = sum(diff_lower.count(p) for p in error_patterns)
    
    # Authentication/authorization code
    auth_patterns = ['auth', 'login', 'password', 'credential', 'token',
                     'session', 'permission', 'access', 'role', 'privilege']
    auth_changed = sum(diff_lower.count(p) for p in auth_patterns)
    
    # Cryptographic code
    crypto_patterns = ['encrypt', 'decrypt', 'hash', 'hmac', 'sign', 
                       'verify', 'ssl', 'tls', 'cipher', 'key', 'iv', 'nonce']
    crypto_changed = sum(diff_lower.count(p) for p in crypto_patterns)
    
    # Input handling
    input_patterns = ['input', 'parse', 'decode', 'deserialize', 'read',
                      'recv', 'request', 'param', 'query', 'body']
    input_changed = sum(diff_lower.count(p) for p in input_patterns)
    
    # Format string vulnerabilities
    format_patterns = ['%s', '%d', '%x', '%n', 'printf', 'fprintf', 
                       'sprintf', 'format']
    format_count = sum(diff_lower.count(p) for p in format_patterns)
    
    # Pointer arithmetic (C/C++ specific)
    pointer_patterns = ['++', '--', '+=', '-=', '*ptr', '&', '->']
    pointer_arith = sum(diff_text.count(p) for p in pointer_patterns)
    
    # Null check removal (very concerning)
    null_patterns = ['!= null', '!= nil', '!= none', 'is not none', '!== null']
    null_removed = sum(1 for p in null_patterns if f'-{p}' in diff_lower or f'- {p}' in diff_lower)
    
    # Line statistics
    added_lines = diff_text.count('\n+') - diff_text.count('\n+++')
    removed_lines = diff_text.count('\n-') - diff_text.count('\n---')
    total_lines = added_lines + removed_lines
    
    # Complexity indicator (nested brackets, long lines)
    complexity = diff_text.count('{') + diff_text.count('}') + diff_text.count('if') + diff_text.count('else')
    
    return {
        'dangerous_function_count': dangerous_count,
        'memory_op_count': memory_count,
        'bounds_check_removed': bounds_removed,
        'error_handling_changed': error_changed,
        'auth_code_changed': auth_changed,
        'crypto_code_changed': crypto_changed,
        'input_handling_changed': input_changed,
        'format_string_count': format_count,
        'pointer_arithmetic': pointer_arith,
        'null_check_removed': null_removed,
        'added_lines_ratio': added_lines / max(total_lines, 1),
        'removed_lines_ratio': removed_lines / max(total_lines, 1),
        'complexity_indicator': complexity,
    }


def compute_weighted_anomaly_score(X, feature_names, feature_weights, contamination=0.15):
    """
    Compute anomaly scores using weighted features.
    
    This method applies feature weights before Isolation Forest,
    giving more importance to vulnerability-indicative features.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        feature_names: List of feature names
        feature_weights: Dictionary mapping feature names to weights
        contamination: Expected anomaly ratio
        
    Returns:
        Tuple of (predictions, decision_scores)
    """
    # Apply weights
    weights = np.array([feature_weights.get(f, 1.0) for f in feature_names])
    X_weighted = X * weights
    
    # Scale weighted features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_weighted)
    
    # Fit Isolation Forest
    model = IsolationForest(
        contamination=contamination,
        n_estimators=200,  # More trees for stability
        max_samples='auto',
        random_state=42,
        n_jobs=-1
    )
    
    predictions = model.fit_predict(X_scaled)
    scores = model.decision_function(X_scaled)
    
    return predictions, scores


def compute_enhanced_embedding_features(embeddings, df, feature_names):
    """
    Enhanced embedding feature extraction using multiple strategies.
    
    Strategy: Instead of raw outlier detection on embeddings (which doesn't work),
    we combine multiple complementary signals:
    
    1. Variance-based: Commits with high variance in embedding space (unusual code)
    2. Security-modulated: Weight embedding distance by security indicators
    3. Isolation score: Directly use embedding-based Isolation Forest scores
    
    Args:
        embeddings: CodeBERT embedding matrix (n_samples, 768)
        df: DataFrame with commit data
        feature_names: List of available feature names
        
    Returns:
        Enhanced feature matrix
    """
    n_samples = len(embeddings)
    
    # 1. PCA reduction for efficiency
    n_comp = min(50, n_samples - 1, embeddings.shape[1])
    pca = PCA(n_components=n_comp, random_state=42)
    emb_reduced = pca.fit_transform(embeddings)
    
    # 2. Compute variance explained by principal components (code complexity indicator)
    explained_var_ratio = pca.explained_variance_ratio_
    
    # Project each commit and compute reconstruction error
    emb_reconstructed = pca.inverse_transform(emb_reduced)
    reconstruction_error = np.linalg.norm(embeddings - emb_reconstructed, axis=1)
    
    # 3. Compute centroid distance
    centroid = np.mean(emb_reduced, axis=0)
    centroid_dist = np.linalg.norm(emb_reduced - centroid, axis=1)
    
    # 4. Run Isolation Forest directly on reduced embeddings
    from sklearn.ensemble import IsolationForest
    iso = IsolationForest(contamination=0.2, n_estimators=100, random_state=42, n_jobs=-1)
    emb_iso_scores = -iso.fit_predict(emb_reduced)  # -1 for anomaly -> 2, 1 for normal -> 0
    emb_decision = iso.decision_function(emb_reduced)
    
    # 5. Build features
    features = []
    
    # Core embedding features
    features.append(reconstruction_error.reshape(-1, 1))
    features.append(centroid_dist.reshape(-1, 1))
    features.append(emb_decision.reshape(-1, 1))  # Most important: IF score on embeddings
    
    # 6. Create powerful interaction features with security indicators
    security_cols = ['has_security_keyword', 'has_memory_keyword', 'has_fix_keyword',
                    'security_keyword_count', 'security_file_count', 'c_cpp_file_count']
    
    # Sum all security indicators into one signal
    security_signal = np.zeros(n_samples)
    for col in security_cols:
        if col in df.columns:
            vals = df[col].fillna(0).values
            if vals.max() > 0:
                security_signal += vals / (vals.max() + 1e-10)  # Normalize and sum
    
    # Key interaction: anomalous embedding + security keywords = vulnerability
    # This amplifies commits that are BOTH semantically unusual AND touch security code
    security_modulated_dist = centroid_dist * (1 + security_signal)
    features.append(security_modulated_dist.reshape(-1, 1))
    
    # Also modulate IF score
    security_modulated_iso = (-emb_decision) * (1 + security_signal)  # Negate so higher = more anomalous
    features.append(security_modulated_iso.reshape(-1, 1))
    
    # 7. Combine all features
    enhanced_features = np.hstack(features)
    
    return enhanced_features


def run_isolation_forest_with_score_ranking(X, contamination=0.15, return_ranking=True):
    """
    Run Isolation Forest and return score-based ranking.
    
    This allows comparison across different contamination levels by using
    raw anomaly scores rather than binary predictions.
    
    Args:
        X: Feature matrix
        contamination: Expected anomaly ratio for model
        return_ranking: If True, return rank percentiles
        
    Returns:
        Tuple of (predictions, scores, rankings)
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = IsolationForest(
        contamination=contamination,
        n_estimators=200,
        max_samples='auto',
        random_state=42,
        n_jobs=-1
    )
    
    predictions = model.fit_predict(X_scaled)
    scores = model.decision_function(X_scaled)
    
    if return_ranking:
        # Convert scores to rankings (0 = most anomalous, 1 = least anomalous)
        rankings = (scores.argsort().argsort() + 1) / len(scores)
        return predictions, scores, rankings
    
    return predictions, scores


def evaluate_at_contamination_levels(X, target_idx, contamination_levels=None):
    """
    Evaluate detection at multiple contamination levels.
    
    Args:
        X: Feature matrix
        target_idx: Index of the target (vulnerable) commit
        contamination_levels: List of contamination values to test
        
    Returns:
        Dictionary with detection results at each level
    """
    if contamination_levels is None:
        contamination_levels = CONTAMINATION_LEVELS
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # First, get raw scores with a middle contamination level
    model = IsolationForest(contamination=0.10, n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_scaled)
    raw_scores = model.decision_function(X_scaled)
    
    # Target score and rank
    target_score = raw_scores[target_idx]
    target_rank = (raw_scores < target_score).sum() + 1  # 1-based rank (1 = most anomalous)
    target_percentile = target_rank / len(raw_scores)
    
    results = {
        'raw_score': target_score,
        'rank': target_rank,
        'percentile': target_percentile,
        'total_commits': len(X),
        'detected_at': {}
    }
    
    # Check detection at each contamination level
    for contam in contamination_levels:
        n_anomalies = int(len(X) * contam)
        detected = target_rank <= n_anomalies
        results['detected_at'][contam] = detected
    
    return results


def detect_anomalies_with_embeddings(df, repo_path, contamination=0.10):
    """
    Enhanced anomaly detection using code embeddings in addition to statistical features.
    
    This demonstrates how semantic code understanding can improve vulnerability detection
    by capturing patterns in the actual code changes, not just metadata.
    
    Args:
        df: DataFrame with commit data
        repo_path: Path to repository for extracting diffs
        contamination: Expected anomaly rate
        
    Returns:
        DataFrame with enhanced anomaly detection
    """
    global _last_model_info
    
    # Load embedding model
    tokenizer, model = get_code_embedding_model()
    
    if tokenizer is None:
        print("Falling back to statistical features only")
        return detect_statistical_anomalies(df, contamination=contamination, use_security_features=True)
    
    print("Computing code embeddings for commits...")
    embeddings = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Computing embeddings"):
        diff = get_commit_diff(repo_path, row['hash'])
        if diff:
            emb = compute_code_embedding(diff, tokenizer, model)
            if emb is not None:
                embeddings.append(emb)
            else:
                embeddings.append(np.zeros(768))  # CodeBERT dimension
        else:
            embeddings.append(np.zeros(768))
    
    embedding_matrix = np.array(embeddings)
    
    # Reduce embedding dimensions using PCA for efficiency
    from sklearn.decomposition import PCA
    pca = PCA(n_components=50, random_state=42)
    embedding_reduced = pca.fit_transform(embedding_matrix)
    
    # Add reduced embeddings as features
    for i in range(50):
        df[f'code_emb_{i}'] = embedding_reduced[:, i]
    
    # Combine with statistical features
    stat_features = [
        'hour_of_day', 'lines_inserted', 'lines_deleted', 'msg_length', 
        'msg_entropy', 'churn_ratio', 'merge_latency_sec', 'is_holiday',
        'test_ratio', 'sensitive_ratio', 'files_changed',
        'is_weekend', 'is_late_night', 'security_keyword_count',
        'has_fix_keyword', 'has_security_keyword', 'has_memory_keyword',
        'security_file_ratio', 'c_cpp_file_ratio', 'config_file_ratio',
        'security_file_count', 'c_cpp_file_count'
    ]
    
    emb_features = [f'code_emb_{i}' for i in range(50)]
    all_features = stat_features + emb_features
    
    # Filter to available features
    available = [f for f in all_features if f in df.columns]
    print(f"Using {len(available)} features ({len(stat_features)} statistical + {len(emb_features)} embedding)")
    
    X = df[available].fillna(0)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model_iso = IsolationForest(contamination=contamination, random_state=42)
    df['anomaly_score'] = model_iso.fit_predict(X_scaled)
    df['isolation_score'] = model_iso.decision_function(X_scaled)
    df['is_anomaly'] = df['anomaly_score'].apply(lambda x: True if x == -1 else False)
    
    _last_model_info = {
        'model': model_iso,
        'scaler': scaler,
        'features': available,
        'X_scaled': X_scaled,
        'threshold': model_iso.offset_,
        'uses_embeddings': True
    }
    
    return df


# ==========================================
# 6. HELPER FUNCTIONS FOR VULNERABILITY DETECTION
# ==========================================

# Repository path mappings for existing repos
REPO_PATH_MAP = {
    'activemq': './activemq', 'archiva': './archiva', 'bc-java': './bc-java',
    'camel': './camel', 'commons-beanutils': './commons-beanutils',
    'commons-imaging': './commons-imagin', 'cryptacular': './cryptacular',
    'cxf': './cxf', 'hbase': './hbase', 'ignite': './ignite',
    'jackson-databind': './jackson-databind', 'kylin': './kylin',
    'lucene-solr': './lucene-solr', 'okhttp': './okhttp',
    'OpenRefine': './OpenRefine', 'openssl': './openssl',
    'paho.mqtt.java': './paho.mqtt.java', 'shiro': './shiro',
    'spring-framework': './spring-framework', 'spring-security': './spring-security',
    'struts': './struts', 'tomcat': './tomca', 'tomee': './tomee',
    'transport-http': './transport-http', 'uaa': './uaa',
    'undertow': './undertow', 'wicket': './wicke',
}

def get_local_repo_path(repo_url):
    """Get local path for a repository URL if it exists."""
    for name, path in REPO_PATH_MAP.items():
        if name.lower() in repo_url.lower():
            if os.path.exists(path):
                return path
    return None


# Feature definitions
BASE_FEATURES = [
    'hour_of_day', 'lines_inserted', 'lines_deleted', 'msg_length', 
    'msg_entropy', 'churn_ratio', 'merge_latency_sec', 'is_holiday',
    'test_ratio', 'sensitive_ratio', 'files_changed'
]

# Security features: Only the most discriminative ones
# Adding too many features can hurt Isolation Forest (curse of dimensionality)
SECURITY_FEATURES = BASE_FEATURES + [
    'has_security_keyword', 'has_memory_keyword', 'security_keyword_count',
    'c_cpp_file_count', 'security_file_count'
]

# Feature weights for enhanced vulnerability detection
# Higher weights = more important for detecting vulnerabilities
# Key insight: Balance between security-specific features and change magnitude
SECURITY_FEATURE_WEIGHTS = {
    # High-signal features (directly related to vulnerabilities)
    'has_security_keyword': 2.5,
    'has_memory_keyword': 2.5,
    'security_keyword_count': 2.0,
    'security_file_ratio': 2.0,
    'security_file_count': 1.8,
    'c_cpp_file_ratio': 1.8,
    'c_cpp_file_count': 1.5,
    'sensitive_ratio': 1.8,
    
    # Medium-signal features (change characteristics that correlate with vulns)
    'has_fix_keyword': 1.5,
    'churn_ratio': 1.8,           # Important: high churn often indicates complex changes
    'lines_inserted': 1.5,        # Keep these reasonably weighted
    'lines_deleted': 1.5,
    'config_file_ratio': 1.3,
    'files_changed': 1.2,
    
    # Lower-signal features (timing - less reliable)
    'is_weekend': 0.8,
    'is_late_night': 0.8,
    'is_holiday': 0.6,
    'hour_of_day': 0.6,
    'msg_length': 0.8,
    'msg_entropy': 0.8,
    'test_ratio': 1.0,           # Neutral - test files aren't vulnerability indicators
    'merge_latency_sec': 0.7,
}

# Contamination levels for multi-threshold evaluation
CONTAMINATION_LEVELS = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]


def print_comparison_table(cve_results, use_codebert=True, contamination=0.15):
    """
    Print a formatted comparison table of detection results.
    
    Args:
        cve_results: List of per-CVE result dictionaries (already sorted)
        use_codebert: Whether CodeBERT results are included
        contamination: Contamination rate used
    """
    # Table header
    print("=" * 70)
    print(f"  DETECTION COMPARISON TABLE (contamination={contamination*100:.0f}%)")
    print("=" * 70)
    
    if use_codebert:
        header = f"{'CVE':<20} {'Base':^10} {'Security':^10} {'CodeBERT':^10} {'Total':^8}"
        separator = "-" * 70
    else:
        header = f"{'CVE':<20} {'Base':^10} {'Security':^10} {'Total':^8}"
        separator = "-" * 58
    
    print(header)
    print(separator)
    
    # Print each CVE row
    for result in cve_results:
        cve = result['cve']
        base = "✅" if result['base'] else "❌"
        security = "✅" if result['security'] else "❌"
        
        if use_codebert:
            codebert = "✅" if result['codebert'] else "❌"
            total = result['detection_count']
            row = f"{cve:<20} {base:^10} {security:^10} {codebert:^10} {total:^8}"
        else:
            total = sum([result['base'], result['security']])
            row = f"{cve:<20} {base:^10} {security:^10} {total:^8}"
        
        print(row)
    
    print(separator)
    
    # Summary row with totals
    base_total = sum(1 for r in cve_results if r['base'])
    security_total = sum(1 for r in cve_results if r['security'])
    n_total = len(cve_results)
    
    if n_total == 0:
        print("No CVEs were successfully analyzed.")
        print("=" * 70)
        return
    
    if use_codebert:
        codebert_total = sum(1 for r in cve_results if r['codebert'])
        summary = f"{'TOTAL DETECTED':<20} {base_total:^10} {security_total:^10} {codebert_total:^10} {'':<8}"
        pct_row = f"{'DETECTION RATE':<20} {base_total/n_total*100:^9.1f}% {security_total/n_total*100:^9.1f}% {codebert_total/n_total*100:^9.1f}% {'':<8}"
    else:
        summary = f"{'TOTAL DETECTED':<20} {base_total:^10} {security_total:^10} {'':<8}"
        pct_row = f"{'DETECTION RATE':<20} {base_total/n_total*100:^9.1f}% {security_total/n_total*100:^9.1f}% {'':<8}"
    
    print(summary)
    print(pct_row)
    print("=" * 70)


def print_enhanced_comparison_table(cve_results, use_codebert=True, contamination=0.15):
    """
    Print enhanced comparison table with rankings and percentiles.
    
    Args:
        cve_results: List of per-CVE result dictionaries
        use_codebert: Whether embedding results are included
        contamination: Contamination rate used
    """
    print("=" * 120)
    print(f"  ENHANCED DETECTION COMPARISON (contamination={contamination*100:.0f}%)")
    print("=" * 120)
    
    # Header - now includes DBSCAN
    if use_codebert:
        header = f"{'CVE':<18} {'Base(IF)':^11} {'Sec(IF)':^11} {'DBSCAN':^11} {'Embed':^11} {'Hybrid':^11} {'Best':^10}"
    else:
        header = f"{'CVE':<18} {'Base(IF)':^11} {'Sec(IF)':^11} {'DBSCAN':^11} {'Hybrid':^11} {'Best':^10}"
    
    print(header)
    print("-" * 120)
    
    for result in cve_results:
        cve = result['cve'][:17]
        
        base_pct = result.get('base_percentile', 1.0) * 100
        sec_pct = result.get('security_percentile', 1.0) * 100
        dbscan_pct = result.get('dbscan_percentile', 1.0) * 100
        hyb_pct = result.get('hybrid_percentile', 1.0) * 100
        
        base_str = f"{base_pct:5.1f}%{'✅' if result.get('base') else '❌'}"
        sec_str = f"{sec_pct:5.1f}%{'✅' if result.get('security_weighted') else '❌'}"
        dbscan_str = f"{dbscan_pct:5.1f}%{'✅' if result.get('dbscan') else '❌'}"
        hyb_str = f"{hyb_pct:5.1f}%{'✅' if result.get('hybrid') else '❌'}"
        
        if use_codebert:
            emb_pct = result.get('embedding_percentile', 1.0) * 100
            emb_str = f"{emb_pct:5.1f}%{'✅' if result.get('embedding_enhanced') else '❌'}"
            
            # Find best method
            methods = {'Base': base_pct, 'Sec': sec_pct, 'DBSCAN': dbscan_pct, 'Embed': emb_pct, 'Hybrid': hyb_pct}
            best = min(methods, key=methods.get)
            
            row = f"{cve:<18} {base_str:^11} {sec_str:^11} {dbscan_str:^11} {emb_str:^11} {hyb_str:^11} {best:^10}"
        else:
            methods = {'Base': base_pct, 'Sec': sec_pct, 'DBSCAN': dbscan_pct, 'Hybrid': hyb_pct}
            best = min(methods, key=methods.get)
            row = f"{cve:<18} {base_str:^11} {sec_str:^11} {dbscan_str:^11} {hyb_str:^11} {best:^10}"
        
        print(row)
    
    print("-" * 120)
    
    # Summary statistics
    n_total = len(cve_results)
    if n_total == 0:
        print("No CVEs were successfully analyzed.")
        print("=" * 120)
        return
    
    base_detected = sum(1 for r in cve_results if r.get('base'))
    sec_detected = sum(1 for r in cve_results if r.get('security_weighted'))
    dbscan_detected = sum(1 for r in cve_results if r.get('dbscan'))
    hyb_detected = sum(1 for r in cve_results if r.get('hybrid'))
    
    base_mean_pct = np.mean([r.get('base_percentile', 1.0) for r in cve_results]) * 100
    sec_mean_pct = np.mean([r.get('security_percentile', 1.0) for r in cve_results]) * 100
    dbscan_mean_pct = np.mean([r.get('dbscan_percentile', 1.0) for r in cve_results]) * 100
    hyb_mean_pct = np.mean([r.get('hybrid_percentile', 1.0) for r in cve_results]) * 100
    
    if use_codebert:
        emb_detected = sum(1 for r in cve_results if r.get('embedding_enhanced'))
        emb_mean_pct = np.mean([r.get('embedding_percentile', 1.0) for r in cve_results]) * 100
        
        print(f"{'DETECTED':<18} {base_detected:^11} {sec_detected:^11} {dbscan_detected:^11} {emb_detected:^11} {hyb_detected:^11}")
        print(f"{'RECALL':<18} {base_detected/n_total*100:^10.1f}% {sec_detected/n_total*100:^10.1f}% {dbscan_detected/n_total*100:^10.1f}% {emb_detected/n_total*100:^10.1f}% {hyb_detected/n_total*100:^10.1f}%")
        print(f"{'MEAN PERCENTILE':<18} {base_mean_pct:^10.1f}% {sec_mean_pct:^10.1f}% {dbscan_mean_pct:^10.1f}% {emb_mean_pct:^10.1f}% {hyb_mean_pct:^10.1f}%")
    else:
        print(f"{'DETECTED':<18} {base_detected:^11} {sec_detected:^11} {dbscan_detected:^11} {hyb_detected:^11}")
        print(f"{'RECALL':<18} {base_detected/n_total*100:^10.1f}% {sec_detected/n_total*100:^10.1f}% {dbscan_detected/n_total*100:^10.1f}% {hyb_detected/n_total*100:^10.1f}%")
        print(f"{'MEAN PERCENTILE':<18} {base_mean_pct:^10.1f}% {sec_mean_pct:^10.1f}% {dbscan_mean_pct:^10.1f}% {hyb_mean_pct:^10.1f}%")
    
    print("=" * 120)
    print("  Lower percentile = better ranking (vulnerability ranked closer to top anomalies)")
    print("  IF = Isolation Forest | DBSCAN = Density-Based Spatial Clustering")
    print("=" * 120)


def print_multi_threshold_analysis(multi_thresh_results, n_total, use_codebert=True):
    """
    Print analysis of detection rates at multiple contamination thresholds.
    
    This shows how each method performs as we vary the anomaly threshold,
    which is key for comparing methods fairly.
    """
    print("\n")
    print("=" * 110)
    print("  MULTI-THRESHOLD ANALYSIS (Detection Rate at Various Contamination Levels)")
    print("=" * 110)
    
    if use_codebert:
        header = f"{'Contamination':^15} {'Base(IF)':^11} {'Sec(IF)':^11} {'DBSCAN':^11} {'Embed':^11} {'Hybrid':^11}"
    else:
        header = f"{'Contamination':^15} {'Base(IF)':^11} {'Sec(IF)':^11} {'DBSCAN':^11} {'Hybrid':^11}"
    
    print(header)
    print("-" * 110)
    
    for level in CONTAMINATION_LEVELS:
        base_rate = multi_thresh_results[level]['base'] / n_total * 100 if n_total > 0 else 0
        sec_rate = multi_thresh_results[level]['security_weighted'] / n_total * 100 if n_total > 0 else 0
        dbscan_rate = multi_thresh_results[level]['dbscan'] / n_total * 100 if n_total > 0 else 0
        hyb_rate = multi_thresh_results[level]['hybrid'] / n_total * 100 if n_total > 0 else 0
        
        if use_codebert:
            emb_rate = multi_thresh_results[level]['embedding_enhanced'] / n_total * 100 if n_total > 0 else 0
            row = f"{level*100:^14.0f}% {base_rate:^10.1f}% {sec_rate:^10.1f}% {dbscan_rate:^10.1f}% {emb_rate:^10.1f}% {hyb_rate:^10.1f}%"
        else:
            row = f"{level*100:^14.0f}% {base_rate:^10.1f}% {sec_rate:^10.1f}% {dbscan_rate:^10.1f}% {hyb_rate:^10.1f}%"
        
        print(row)
    
    print("-" * 110)
    
    # Calculate AUC-like metric (area under the detection rate curve)
    print("\n  📊 Performance Metrics (higher is better):")
    
    # Compute normalized area under the curve for each method
    levels = np.array(CONTAMINATION_LEVELS)
    
    for method in ['base', 'security_weighted', 'dbscan', 'embedding_enhanced', 'hybrid']:
        if method == 'embedding_enhanced' and not use_codebert:
            continue
        
        rates = np.array([multi_thresh_results[l][method] / n_total for l in CONTAMINATION_LEVELS])
        
        # Trapezoidal integration normalized by max possible area
        auc = np.trapz(rates, levels) / (levels[-1] - levels[0])
        
        # Mean rank (lower is better, invert for display)
        mean_rate = np.mean(rates) * 100
        
        method_name = {
            'base': 'Base (IF)',
            'security_weighted': 'Security (IF)',
            'dbscan': 'DBSCAN',
            'embedding_enhanced': 'Embedding',
            'hybrid': 'Hybrid'
        }[method]
        
        print(f"     {method_name:15s}: AUC={auc:.3f}, Mean Detection Rate={mean_rate:.1f}%")
    
    print("=" * 90)


def print_aggregate_metrics(results, contamination, use_codebert=True):
    """
    Print aggregate performance metrics including precision-like metrics.
    """
    print("\n")
    print("=" * 80)
    print(f"  AGGREGATE PERFORMANCE METRICS (at {contamination*100:.0f}% contamination)")
    print("=" * 80)
    
    metrics = {}
    
    for method, data in results.items():
        if method == 'embedding_enhanced' and not use_codebert:
            continue
        
        if not data['detected']:
            continue
        
        detected = sum(data['detected'])
        total = len(data['detected'])
        
        # Recall = proportion of vulnerabilities detected
        recall = detected / total if total > 0 else 0
        
        # Mean percentile (lower is better - vuln ranked higher)
        mean_pct = np.mean(data['percentiles']) if data['percentiles'] else 1.0
        
        # Median rank
        median_rank = np.median(data['ranks']) if data['ranks'] else 0
        
        # "Precision" approximation: assuming contamination% are flagged,
        # what fraction of true vulns are in the flagged set
        # This is essentially the recall at the given contamination level
        precision_approx = recall  # At contamination level, this equals recall for single-vuln-per-repo case
        
        # F1-like score
        f1 = 2 * (precision_approx * recall) / (precision_approx + recall) if (precision_approx + recall) > 0 else 0
        
        metrics[method] = {
            'detected': detected,
            'total': total,
            'recall': recall,
            'mean_percentile': mean_pct,
            'median_rank': median_rank,
            'f1_approx': f1
        }
    
    # Display
    method_names = {
        'base': 'Base Features (IF)',
        'security_weighted': 'Security Weighted (IF)',
        'dbscan': 'DBSCAN Clustering',
        'embedding_enhanced': 'Embedding Enhanced',
        'hybrid': 'Hybrid Ensemble'
    }
    
    for method, m in metrics.items():
        name = method_names.get(method, method)
        bar = "█" * int(m['recall'] * 20) + "░" * (20 - int(m['recall'] * 20))
        
        print(f"\n  {name}:")
        print(f"     Detected:        {m['detected']:3d}/{m['total']} ({m['recall']*100:5.1f}%) {bar}")
        print(f"     Mean Percentile: {m['mean_percentile']*100:5.1f}% (lower = better)")
        print(f"     Median Rank:     {m['median_rank']:5.0f}")
    
    print("\n" + "-" * 80)
    
    # Winner determination
    if metrics:
        best_recall = max(metrics.items(), key=lambda x: x[1]['recall'])
        best_ranking = min(metrics.items(), key=lambda x: x[1]['mean_percentile'])
        
        print(f"  🏆 Best Recall:    {method_names[best_recall[0]]} ({best_recall[1]['recall']*100:.1f}%)")
        print(f"  🏆 Best Ranking:   {method_names[best_ranking[0]]} (mean {best_ranking[1]['mean_percentile']*100:.1f}% percentile)")
    
    print("=" * 80)
    print("  IF = Isolation Forest | DBSCAN = Density-Based Spatial Clustering")
    print("=" * 80)


def plot_multi_threshold_curves(multi_thresh_results, n_total, use_codebert=True):
    """
    Generate visualization of detection rates across contamination levels.
    
    Creates a line plot showing how each method's detection rate changes
    as we vary the contamination threshold.
    """
    if not PLOTLY_AVAILABLE:
        print("⚠️ Plotly not available for visualization")
        return None
    
    fig = go.Figure()
    
    methods = ['base', 'security_weighted', 'dbscan', 'hybrid']
    if use_codebert:
        methods.insert(3, 'embedding_enhanced')
    
    colors = {
        'base': '#3498db',
        'security_weighted': '#e74c3c',
        'dbscan': '#f39c12',
        'embedding_enhanced': '#9b59b6',
        'hybrid': '#2ecc71'
    }
    
    names = {
        'base': 'Base Features (IF)',
        'security_weighted': 'Security Weighted (IF)',
        'dbscan': 'DBSCAN Clustering',
        'embedding_enhanced': 'Embedding Enhanced',
        'hybrid': 'Hybrid Ensemble'
    }
    
    for method in methods:
        rates = [multi_thresh_results[l][method] / n_total * 100 for l in CONTAMINATION_LEVELS]
        levels_pct = [l * 100 for l in CONTAMINATION_LEVELS]
        
        fig.add_trace(go.Scatter(
            x=levels_pct,
            y=rates,
            mode='lines+markers',
            name=names[method],
            line=dict(color=colors[method], width=3),
            marker=dict(size=10)
        ))
    
    # Add diagonal reference line (random baseline)
    fig.add_trace(go.Scatter(
        x=[l * 100 for l in CONTAMINATION_LEVELS],
        y=[l * 100 for l in CONTAMINATION_LEVELS],
        mode='lines',
        name='Random Baseline',
        line=dict(color='gray', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title='Detection Rate vs Contamination Threshold (IF vs DBSCAN)',
        xaxis_title='Contamination Level (%)',
        yaxis_title='Detection Rate (%)',
        template='plotly_white',
        width=900,
        height=600,
        legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99)
    )
    
    fig.show()
    return fig


# ==========================================
# EXAMPLE USAGE
# ==========================================

if __name__ == "__main__":
    # Run enhanced vulnerability detection analysis comparing four methods:
    #
    # Methods compared:
    #   1. Base Features: Basic statistical features (lines changed, entropy, etc.)
    #   2. Security (Weighted): Security features with vulnerability-specific weights
    #   3. Embedding (Enhanced): CodeBERT embeddings with KNN-based outlier detection
    #   4. Hybrid Ensemble: Score fusion of all methods
    #
    # Key improvements over baseline:
    #   - Feature weighting: Security-related features weighted higher
    #   - Enhanced embeddings: KNN distance + centroid distance features
    #   - Multi-threshold evaluation: Compare at multiple contamination levels
    #   - Rank-based metrics: Mean percentile and median rank for fair comparison
    #
    # Options:
    #   n_samples: Number of vulnerabilities to analyze (None = all)
    #   n_commits: Context window size for each vulnerability
    #   contamination: Primary anomaly rate threshold for binary detection
    #   cached_only: Only use repos with fully cached data (fastest)
    #   local_repos_only: Only use repos already cloned locally
    #   use_codebert: Include CodeBERT embedding method (requires transformers)
    #   multi_threshold: Evaluate at multiple contamination levels
    #   visualize: Generate detailed plots per CVE
    
    results = run_vulnerability_detection(
        n_samples=None,          # Analyze all available vulnerabilities
        n_commits=2000,           # Use 400 commits as context window
        contamination=0.15,      # Primary anomaly threshold (15%)
        cached_only=True,        # Only use repos with fully cached data (fastest)
        local_repos_only=False,   # Fallback: use already-cloned repos
        use_codebert=True,       # Include enhanced embedding comparison
        multi_threshold=True,    # Evaluate at multiple contamination levels
        visualize=False,          # Set to True for detailed per-CVE visualizations
        # disable_cache=True       # Set to True to disable caching
    )
    # results = run_vulnerability_detection(
    #     specific_repo_path="./git_repos/xz",
    #     specific_repo_url="https://github.com/tukaani-project/xz.git",
    #     specific_commit='cf44e4b',
    #     n_commits=500,
    #     contamination=0.15,
    #     use_codebert=False,
    #     visualize=True  # Enable visualizations
    # )
    # visualize_isolation_forest_decision_path(
    #     model=_last_model_info['model'],
    #     X=_last_model_info['X_scaled'],
    #     feature_names=_last_model_info['features'],
    #     sample_idx=0,  # Index of commit to explain
    #     max_depth=4    # How deep to visualize (4 layers)
    # )
    # Specify features to visualize
    # result = explore_commit_features(
    #     repo_path="./git_repos/pandas",
    #     repo_url="https://github.com/pandas-dev/pandas.git",
    #     features=['hour_of_day', 'day_of_week', 'churn_ratio', 'msg_length', 'lines_inserted', 'lines_deleted', 'files_changed'],
    #     n_commits=10000
    # )
    # # Generate multi-threshold visualization
    # if results.get('multi_threshold') and results.get('cve_results'):
    #     print("\n📊 Generating multi-threshold visualization...")
    #     plot_multi_threshold_curves(
    #         results['multi_threshold'], 
    #         len(results['cve_results']), 
    #         use_codebert=True
    #     )
    
    print("\n✅ Enhanced vulnerability detection analysis complete!")
    print("\n📊 Detection comparison complete!")