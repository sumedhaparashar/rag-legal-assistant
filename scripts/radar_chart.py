# scripts/radar_chart.py
"""
Professional Radar Chart for Research Paper
Based on YOUR ACTUAL evaluation data
"""

import matplotlib.pyplot as plt
import numpy as np
from math import pi

# ============================================
# YOUR ACTUAL DATA
# ============================================

metrics = ['Cosine', 'Euclidean', 'Hybrid']
colors = ['#E74C3C', '#3498DB', '#2ECC71']

# Raw data
relevance = [0.625, 0.543, 0.948]      # Relevance scores
response_times = [0.141, 0.013, 0.034]  # Seconds
citations = [10.3, 9.7, 9.3]            # Citations per answer
coverage = [33.0, 33.0, 33.0]           # Coverage percentage

# Normalize data to 0-1 scale (higher = better)
# For response times: invert (lower time = higher score)
max_time = max(response_times)
speed_score = [1 - (t/max_time) for t in response_times]

# Normalize citations (max theoretical = 15)
citation_score = [c/15 for c in citations]

# Coverage is already percentage, divide by 100
coverage_score = [c/100 for c in coverage]

# Calculate efficiency (relevance / time) and normalize
efficiency_raw = [relevance[i] / response_times[i] for i in range(3)]
max_efficiency = max(efficiency_raw)
efficiency_score = [e/max_efficiency for e in efficiency_raw]

# ============================================
# Create Radar Chart
# ============================================

# Define categories
categories = ['Relevance\nScore', 'Speed\n(Faster=Better)', 
              'Citations', 'Coverage', 'Efficiency']
num_vars = len(categories)

# Calculate angles for each category
angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
angles += angles[:1]  # Close the loop

# Data for each metric
cosine_values = [relevance[0], speed_score[0], citation_score[0], coverage_score[0], efficiency_score[0]]
cosine_values += cosine_values[:1]

euclidean_values = [relevance[1], speed_score[1], citation_score[1], coverage_score[1], efficiency_score[1]]
euclidean_values += euclidean_values[:1]

hybrid_values = [relevance[2], speed_score[2], citation_score[2], coverage_score[2], efficiency_score[2]]
hybrid_values += hybrid_values[:1]

# Create figure
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

# Plot each metric
ax.plot(angles, cosine_values, 'o-', linewidth=2, label='Cosine', color='#E74C3C', markersize=8)
ax.fill(angles, cosine_values, alpha=0.1, color='#E74C3C')

ax.plot(angles, euclidean_values, 'o-', linewidth=2, label='Euclidean', color='#3498DB', markersize=8)
ax.fill(angles, euclidean_values, alpha=0.1, color='#3498DB')

ax.plot(angles, hybrid_values, 'o-', linewidth=3, label='Hybrid', color='#2ECC71', markersize=10)
ax.fill(angles, hybrid_values, alpha=0.2, color='#2ECC71')

# Customize the chart
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
ax.set_title('Performance Radar Chart: Cosine vs Euclidean vs Hybrid\n(Higher Values = Better Performance)', 
             fontsize=14, fontweight='bold', pad=30)
ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), fontsize=11)
ax.grid(True, alpha=0.3)

# Add a circle at 0.5 (midpoint)
circle = plt.Circle((0, 0), 0.5, transform=ax.transData._b, 
                    fill=False, edgecolor='gray', linestyle='--', alpha=0.5)
ax.add_artist(circle)

plt.tight_layout()
plt.savefig('research_results/radar_chart.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================
# Print the data for verification
# ============================================
print("\n" + "="*60)
print("RADAR CHART DATA (Normalized 0-1 scale)")
print("="*60)
print(f"\n{'Metric':<12} {'Relevance':<10} {'Speed':<8} {'Citations':<10} {'Coverage':<10} {'Efficiency':<10}")
print("-"*60)
print(f"{'Cosine':<12} {relevance[0]:<10.3f} {speed_score[0]:<8.3f} {citation_score[0]:<10.3f} {coverage_score[0]:<10.3f} {efficiency_score[0]:<10.3f}")
print(f"{'Euclidean':<12} {relevance[1]:<10.3f} {speed_score[1]:<8.3f} {citation_score[1]:<10.3f} {coverage_score[1]:<10.3f} {efficiency_score[1]:<10.3f}")
print(f"{'Hybrid':<12} {relevance[2]:<10.3f} {speed_score[2]:<8.3f} {citation_score[2]:<10.3f} {coverage_score[2]:<10.3f} {efficiency_score[2]:<10.3f}")
print("\n✅ Radar chart saved to: research_results/radar_chart.png")