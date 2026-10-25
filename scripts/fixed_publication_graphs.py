# scripts/fixed_publication_graphs.py
"""
Fixed publication graphs with proper sizing for research paper
"""

import matplotlib.pyplot as plt
import numpy as np

# Set proper parameters for clear display
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 11,
    'figure.dpi': 150,  # Lower DPI for better display on screen
    'savefig.dpi': 300,  # High DPI for printing
    'figure.figsize': (10, 6),  # Widescreen format
    'figure.facecolor': 'white',
    'axes.facecolor': '#f8f9fa'
})

# YOUR ACTUAL DATA
metrics = ['Cosine', 'Euclidean', 'Hybrid']
relevance_scores = [0.625, 0.543, 0.948]
relevance_std = [0.15, 0.18, 0.12]
retrieval_times = [0.141, 0.013, 0.034]
time_std = [0.052, 0.004, 0.009]
citations = [10.3, 9.7, 9.3]
coverage = [33.0, 33.0, 33.0]
colors = ['#E74C3C', '#3498DB', '#2ECC71']  # Professional colors

print("Generating publication-quality graphs...")

# Figure 1: Relevance Score Comparison
fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.bar(metrics, relevance_scores, yerr=relevance_std, 
              capsize=8, color=colors, alpha=0.8, 
              edgecolor='black', linewidth=1.5, error_kw={'linewidth': 2})

# Add value labels
for bar, score, std in zip(bars, relevance_scores, relevance_std):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
            f'{score:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=13)

ax.set_ylabel('Relevance Score (0-1 scale)', fontsize=14, fontweight='bold')
ax.set_xlabel('Similarity Metric', fontsize=14, fontweight='bold')
ax.set_title('Semantic Relevance by Retrieval Method\n(Higher is Better)', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_ylim(0, 1.15)
ax.set_xticklabels(metrics, fontsize=13)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add horizontal line at Euclidean baseline
ax.axhline(y=0.543, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
ax.text(2.5, 0.55, 'Euclidean Baseline', fontsize=10, style='italic', color='gray')

plt.tight_layout()
plt.savefig('research_results/figure1_relevance.png', dpi=300, bbox_inches='tight')
print("✓ Figure 1 saved: figure1_relevance.png")

# Figure 2: Retrieval Times
fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.bar(metrics, retrieval_times, yerr=time_std, 
              capsize=8, color=colors, alpha=0.8,
              edgecolor='black', linewidth=1.5, error_kw={'linewidth': 2})

for bar, time_val, std in zip(bars, retrieval_times, time_std):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{time_val:.3f}s', ha='center', va='bottom', fontweight='bold', fontsize=13)

ax.set_ylabel('Retrieval Time (seconds)', fontsize=14, fontweight='bold')
ax.set_xlabel('Similarity Metric', fontsize=14, fontweight='bold')
ax.set_title('Response Time by Retrieval Method\n(Lower is Better)', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xticklabels(metrics, fontsize=13)
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('research_results/figure2_times.png', dpi=300, bbox_inches='tight')
print("✓ Figure 2 saved: figure2_times.png")

# Figure 3: Coverage (Identical Finding)
fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.bar(metrics, coverage, color=colors, alpha=0.8,
              edgecolor='black', linewidth=1.5)

for bar, cov in zip(bars, coverage):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{cov:.0f}%', ha='center', va='bottom', fontweight='bold', fontsize=13)

ax.set_ylabel('Keyword Coverage (%)', fontsize=14, fontweight='bold')
ax.set_xlabel('Similarity Metric', fontsize=14, fontweight='bold')
ax.set_title('Keyword Coverage: Identical Across All Metrics\n(Coverage Doesn\'t Capture Semantic Quality)', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_ylim(0, 50)
ax.set_xticklabels(metrics, fontsize=13)
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('research_results/figure3_coverage.png', dpi=300, bbox_inches='tight')
print("✓ Figure 3 saved: figure3_coverage.png")

print("\n✅ All graphs generated successfully!")
print("📁 Check the 'research_results' folder for PNG files")