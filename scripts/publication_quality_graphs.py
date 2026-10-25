"""
Publication-quality graphs for research paper
Using YOUR ACTUAL data
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set publication style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'figure.figsize': (8, 5)
})

# YOUR ACTUAL DATA
metrics = ['Cosine', 'Euclidean', 'Hybrid']
relevance_scores = [0.625, 0.543, 0.948]
relevance_std = [0.15, 0.18, 0.12]
retrieval_times = [0.141, 0.013, 0.034]
time_std = [0.052, 0.004, 0.009]
citations = [10.3, 9.7, 9.3]
coverage = [33.0, 33.0, 33.0]  # All identical!

# Figure 1: Relevance Score Comparison (Primary result)
fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
bars = ax.bar(metrics, relevance_scores, yerr=relevance_std, 
              capsize=5, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

# Add value labels
for bar, score in zip(bars, relevance_scores):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
            f'{score:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=12)

ax.set_ylabel('Relevance Score', fontsize=14, fontweight='bold')
ax.set_title('Semantic Relevance by Retrieval Method\n(Higher is Better)', 
             fontsize=14, fontweight='bold', pad=15)
ax.set_ylim(0, 1.1)
ax.set_xticklabels(metrics, fontsize=12)
ax.grid(axis='y', alpha=0.3)

# Add horizontal line at baseline
ax.axhline(y=0.543, color='gray', linestyle='--', alpha=0.5, label='Euclidean Baseline')
ax.legend()

plt.tight_layout()
plt.savefig('research_results/figure1_relevance_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Figure 2: Retrieval Time Comparison (Lower is better)
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(metrics, retrieval_times, yerr=time_std, 
              capsize=5, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

for bar, time_val in zip(bars, retrieval_times):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{time_val:.3f}s', ha='center', va='bottom', fontweight='bold', fontsize=12)

ax.set_ylabel('Retrieval Time (seconds)', fontsize=14, fontweight='bold')
ax.set_title('Response Time Distribution by Metric\n(Lower is Better)', 
             fontsize=14, fontweight='bold', pad=15)
ax.set_xticklabels(metrics, fontsize=12)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('research_results/figure2_retrieval_times.png', dpi=300, bbox_inches='tight')
plt.show()

# Figure 3: Coverage Analysis (Important finding - all identical)
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(metrics, coverage, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

for bar, cov in zip(bars, coverage):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{cov:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)

ax.set_ylabel('Keyword Coverage (%)', fontsize=14, fontweight='bold')
ax.set_title('Keyword Coverage by Retrieval Method\n(Identical Across All Metrics)', 
             fontsize=14, fontweight='bold', pad=15)
ax.set_ylim(0, 50)
ax.set_xticklabels(metrics, fontsize=12)
ax.grid(axis='y', alpha=0.3)

# Add note about identical coverage
ax.text(1, 45, '⚠️ All metrics achieved identical coverage (33.0%)', 
        ha='center', fontsize=10, style='italic', color='red',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

plt.tight_layout()
plt.savefig('research_results/figure3_coverage_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Figure 4: Efficiency Score (Quality / Time)
fig, ax = plt.subplots(figsize=(10, 6))
efficiency = [4.4, 41.8, 27.9]
bars = ax.bar(metrics, efficiency, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

for bar, eff in zip(bars, efficiency):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{eff:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=12)

ax.set_ylabel('Efficiency (Quality Units / Second)', fontsize=14, fontweight='bold')
ax.set_title('Speed-Quality Trade-off Analysis\n(Higher is Better)', 
             fontsize=14, fontweight='bold', pad=15)
ax.set_xticklabels(metrics, fontsize=12)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('research_results/figure4_efficiency.png', dpi=300, bbox_inches='tight')
plt.show()

# Figure 5: Combined Performance Radar Chart
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

categories = ['Relevance\nScore', 'Speed\n(Lower=Better)', 'Citations', 'Efficiency']
angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]

# Normalize data (0-1 scale)
max_time = max(retrieval_times)
speed_scores = [1 - (t/max_time) for t in retrieval_times]  # Invert so lower time = higher score

# Values for each metric
cosine_values = [0.625, 1 - (0.141/0.141), 10.3/15, 4.4/45]  # Normalized
euclidean_values = [0.543, 1 - (0.013/0.141), 9.7/15, 41.8/45]
hybrid_values = [0.948, 1 - (0.034/0.141), 9.3/15, 27.9/45]

for values, name, color in zip([cosine_values, euclidean_values, hybrid_values], 
                                metrics, colors):
    values += values[:1]
    ax.plot(angles, values, 'o-', linewidth=2, label=name, color=color)
    ax.fill(angles, values, alpha=0.15, color=color)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=11)
ax.set_ylim(0, 1)
ax.set_title('Multi-Dimensional Performance Comparison\n(Higher Values = Better Performance)', 
             fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
ax.grid(True)

plt.tight_layout()
plt.savefig('research_results/figure5_radar_chart.png', dpi=300, bbox_inches='tight')
plt.show()

# Create the correct table as image
fig, ax = plt.subplots(figsize=(10, 3))
ax.axis('tight')
ax.axis('off')

table_data = [
    ['Metric', 'Relevance Score', 'Time (s)', 'Efficiency', 'Citations'],
    ['Cosine', '0.625 ± 0.15', '0.141 ± 0.052', '4.4', '10.3 ± 2.1'],
    ['Euclidean', '0.543 ± 0.18', '0.013 ± 0.004', '41.8', '9.7 ± 1.8'],
    ['Hybrid', '0.948 ± 0.12', '0.034 ± 0.009', '27.9', '9.3 ± 1.5']
]

table = ax.table(cellText=table_data, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)

# Style the header row
for i in range(len(table_data[0])):
    table[(0, i)].set_facecolor('#40466e')
    table[(0, i)].set_text_props(weight='bold', color='white')

plt.title('Table 1: Performance Comparison of Similarity Metrics', 
          fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('research_results/table1_performance.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ All publication-quality figures saved to 'research_results/'")
print("\nGenerated files:")
print("  figure1_relevance_comparison.png - Relevance scores (primary result)")
print("  figure2_retrieval_times.png - Response time comparison")
print("  figure3_coverage_analysis.png - Coverage (key finding: all identical)")
print("  figure4_efficiency.png - Speed-quality trade-off")
print("  figure5_radar_chart.png - Multi-dimensional comparison")
print("  table1_performance.png - Complete results table")