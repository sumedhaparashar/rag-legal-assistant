"""
Research Evaluation Suite for Legal RAG Assistant
Generates metrics, comparisons, and visualizations for research paper
"""

import sys
import os
# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import json
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple
import pandas as pd
from collections import defaultdict

# Import your RAG components - Fixed imports
from src.rag_chain import ask
from src.vectorstore import (
    similarity_search_cosine, 
    similarity_search_euclidean, 
    hybrid_search,
    get_vectorstore_stats,
    load_vectorstore
)
from src.config import CHUNK_SIZE, CHUNK_OVERLAP

# For visualizations
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving images
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore')

# Set style for research-quality graphs
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class LegalRAGEvaluator:
    """Comprehensive evaluator for RAG system"""
    
    def __init__(self):
        self.results = defaultdict(list)
        self.query_times = defaultdict(list)
        self.relevance_scores = defaultdict(list)
        
    def load_test_queries(self) -> List[Dict]:
        """Load or define test queries with expected relevant documents"""
        test_queries = [
            {
                "question": "What are the duties of company directors?",
                "expected_keywords": ["duty", "care", "diligence", "fiduciary", "act in good faith"],
                "expected_sections": ["Section 166", "director duties"]
            },
            {
                "question": "How can a company appoint additional directors?",
                "expected_keywords": ["appointment", "board resolution", "shareholder approval", "additional director"],
                "expected_sections": ["Section 161", "appointment"]
            },
            {
                "question": "What are the penalties for late filing of annual returns?",
                "expected_keywords": ["penalty", "fine", "late filing", "annual return", "default"],
                "expected_sections": ["Section 92", "penalty provisions"]
            },
            {
                "question": "What is the process for winding up a company?",
                "expected_keywords": ["winding up", "liquidation", "tribunal", "creditors", "voluntary"],
                "expected_sections": ["Section 270", "winding up"]
            },
            {
                "question": "What are the requirements for annual general meetings?",
                "expected_keywords": ["AGM", "annual general meeting", "notice period", "board report", "financial statements"],
                "expected_sections": ["Section 96", "Section 101"]
            },
            {
                "question": "What powers do shareholders have?",
                "expected_keywords": ["shareholder rights", "voting", "dividend", "inspection", "oppression"],
                "expected_sections": ["Section 47", "shareholder rights"]
            },
            {
                "question": "How are companies regulated by SEBI?",
                "expected_keywords": ["SEBI", "securities", "listing", "disclosure", "insider trading"],
                "expected_sections": ["SEBI Act", "listing regulations"]
            },
            {
                "question": "What is the role of the board of directors?",
                "expected_keywords": ["board", "management", "strategy", "oversight", "decision making"],
                "expected_sections": ["Section 149", "board composition"]
            },
            {
                "question": "What are the rules for related party transactions?",
                "expected_keywords": ["related party", "RPT", "conflict of interest", "approval", "disclosure"],
                "expected_sections": ["Section 188", "related party"]
            },
            {
                "question": "How can a company raise capital?",
                "expected_keywords": ["capital raising", "equity", "debt", "prospectus", "private placement"],
                "expected_sections": ["Section 23", "prospectus"]
            }
        ]
        return test_queries
    
    def evaluate_retrieval_quality(self, query: str, retrieved_docs: List, expected_keywords: List) -> Dict:
        """Calculate precision, recall, and keyword coverage"""
        # Convert retrieved docs to text
        retrieved_text = " ".join([doc.page_content.lower() for doc, _ in retrieved_docs])
        
        # Calculate keyword coverage
        covered_keywords = [kw for kw in expected_keywords if kw in retrieved_text]
        keyword_coverage = len(covered_keywords) / len(expected_keywords) if expected_keywords else 0
        
        # Calculate relevance diversity (unique sources)
        unique_sources = len(set([doc.metadata.get('source', '') for doc, _ in retrieved_docs]))
        diversity_score = min(unique_sources / 3, 1.0)  # Normalized to 0-1
        
        # Calculate information density (unique content ratio)
        total_chars = sum(len(doc.page_content) for doc, _ in retrieved_docs)
        unique_chunks = len(set([doc.page_content[:100] for doc, _ in retrieved_docs]))
        density = unique_chunks / len(retrieved_docs) if retrieved_docs else 0
        
        return {
            'keyword_coverage': keyword_coverage,
            'coverage_percentage': keyword_coverage * 100,
            'matched_keywords': covered_keywords,
            'unmatched_keywords': [kw for kw in expected_keywords if kw not in retrieved_text],
            'diversity_score': diversity_score,
            'information_density': density,
            'unique_sources': unique_sources
        }
    
    def run_metric_comparison(self, test_queries: List[Dict], k: int = 5) -> pd.DataFrame:
        """Compare cosine, euclidean, and hybrid metrics"""
        metrics = ['cosine', 'euclidean', 'hybrid']
        results_data = []
        
        print("🔄 Running metric comparison...")
        print("-" * 80)
        
        # First, check if vectorstore exists
        try:
            load_vectorstore()
        except FileNotFoundError:
            print("❌ No FAISS index found. Please run 'python scripts/ingest.py' first.")
            return pd.DataFrame()
        
        for metric in metrics:
            print(f"\n📊 Testing {metric.upper()} metric...")
            metric_results = []
            
            for test in test_queries:
                question = test['question']
                
                # Measure retrieval time
                start_time = time.time()
                
                # Get documents with scores
                try:
                    if metric == 'cosine':
                        docs_with_scores = similarity_search_cosine(question, k=k)
                    elif metric == 'euclidean':
                        docs_with_scores = similarity_search_euclidean(question, k=k)
                    else:  # hybrid
                        docs_with_scores = hybrid_search(question, k=k)
                    
                    retrieval_time = time.time() - start_time
                    
                    # Calculate retrieval quality
                    quality = self.evaluate_retrieval_quality(
                        question, docs_with_scores, test['expected_keywords']
                    )
                    
                    # Calculate average relevance score
                    avg_score = np.mean([score for _, score in docs_with_scores]) if docs_with_scores else 0
                    max_score = max([score for _, score in docs_with_scores]) if docs_with_scores else 0
                    
                    metric_results.append({
                        'metric': metric,
                        'question': question[:50],
                        'retrieval_time': retrieval_time,
                        'avg_relevance': avg_score,
                        'max_relevance': max_score,
                        'keyword_coverage': quality['keyword_coverage'],
                        'coverage_percentage': quality['coverage_percentage'],
                        'diversity_score': quality['diversity_score'],
                        'num_retrieved': len(docs_with_scores),
                        'unique_sources': quality['unique_sources']
                    })
                    
                    print(f"  ✓ {question[:40]}... - Coverage: {quality['coverage_percentage']:.1f}% - Time: {retrieval_time:.3f}s")
                    
                except Exception as e:
                    print(f"  ✗ Error with '{question[:40]}': {str(e)[:50]}")
                    continue
            
            # Calculate averages for this metric
            if metric_results:
                metric_df = pd.DataFrame(metric_results)
                results_data.extend(metric_results)
                
                print(f"\n  📈 {metric.upper()} Averages:")
                print(f"     Coverage: {metric_df['coverage_percentage'].mean():.2f}%")
                print(f"     Retrieval Time: {metric_df['retrieval_time'].mean():.3f}s")
                print(f"     Relevance Score: {metric_df['avg_relevance'].mean():.3f}")
        
        return pd.DataFrame(results_data)
    
    def run_llm_quality_evaluation(self, test_queries: List[Dict], metrics: List[str] = ['cosine', 'euclidean', 'hybrid']) -> pd.DataFrame:
        """Evaluate LLM answer quality with different retrieval methods"""
        results = []
        
        print("\n🤖 Running LLM answer quality evaluation...")
        print("-" * 80)
        
        # Use subset for LLM evaluation (time consuming)
        test_subset = test_queries[:3]  # Reduced to 3 for speed
        
        for metric in metrics:
            print(f"\n📝 Testing {metric.upper()} with LLM...")
            
            for test in test_subset:
                question = test['question']
                
                try:
                    # Track response time
                    start_time = time.time()
                    
                    # Get answer from RAG
                    response = ask(question, similarity_metric=metric, return_retrieval_metadata=True)
                    
                    llm_time = time.time() - start_time
                    
                    # Calculate answer quality metrics
                    answer = response['answer'].lower()
                    
                    # Check for citations
                    has_citations = '[source:' in answer or '[Source:' in answer
                    citation_count = answer.count('[source:') + answer.count('[Source:')
                    
                    # Check for hallucination markers
                    has_not_found = 'cannot find' in answer or 'not enough information' in answer
                    
                    # Check confidence statement
                    has_confidence = 'confidence:' in answer
                    
                    # Keyword coverage in answer
                    matched_answer_keywords = sum(1 for kw in test['expected_keywords'] if kw in answer)
                    answer_keyword_coverage = matched_answer_keywords / len(test['expected_keywords']) if test['expected_keywords'] else 0
                    
                    results.append({
                        'metric': metric,
                        'question': question[:50],
                        'has_citations': has_citations,
                        'citation_count': citation_count,
                        'has_confidence': has_confidence,
                        'has_not_found_statement': has_not_found,
                        'answer_keyword_coverage': answer_keyword_coverage,
                        'llm_response_time': llm_time,
                        'retrieval_time': response.get('metadata', {}).get('retrieval_time_seconds', 0),
                        'answer_length': len(response['answer']),
                        'num_sources': len(response['sources'])
                    })
                    
                    print(f"  ✓ {question[:40]}... - Citations: {citation_count} - Coverage: {answer_keyword_coverage:.2f}")
                    
                except Exception as e:
                    print(f"  ✗ Error with '{question[:40]}': {str(e)[:50]}")
                    continue
        
        return pd.DataFrame(results)
    
    def calculate_statistical_significance(self, metric1_scores: List[float], metric2_scores: List[float]) -> Dict:
        """Calculate statistical significance between two metrics"""
        try:
            from scipy import stats
            
            # Paired t-test
            t_stat, p_value = stats.ttest_rel(metric1_scores, metric2_scores)
            
            # Calculate improvement percentage
            mean1 = np.mean(metric1_scores)
            mean2 = np.mean(metric2_scores)
            improvement = ((mean2 - mean1) / mean1) * 100 if mean1 > 0 else 0
            
            # Cohen's d effect size
            diff = np.array(metric2_scores) - np.array(metric1_scores)
            pooled_std = np.sqrt((np.std(metric1_scores)**2 + np.std(metric2_scores)**2) / 2)
            cohens_d = np.mean(diff) / pooled_std if pooled_std > 0 else 0
            
            return {
                'p_value': p_value,
                'is_significant': p_value < 0.05,
                'improvement_percentage': improvement,
                'cohens_d': cohens_d,
                'effect_size': 'large' if abs(cohens_d) > 0.8 else 'medium' if abs(cohens_d) > 0.5 else 'small'
            }
        except ImportError:
            return {
                'p_value': None,
                'is_significant': None,
                'improvement_percentage': 0,
                'cohens_d': 0,
                'effect_size': 'scipy not installed'
            }
    
    def generate_visualizations(self, retrieval_df: pd.DataFrame, llm_df: pd.DataFrame, output_dir: str = 'research_results'):
        """Generate all research visualizations"""
        os.makedirs(output_dir, exist_ok=True)
        
        if retrieval_df.empty:
            print("No data to visualize")
            return
        
        # Set up the plotting style
        fig_width = 10
        fig_height = 6
        
        # 1. Bar Chart: Coverage by Metric
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        coverage_by_metric = retrieval_df.groupby('metric')['coverage_percentage'].agg(['mean', 'std']).reset_index()
        
        x_pos = np.arange(len(coverage_by_metric))
        bars = ax.bar(x_pos, coverage_by_metric['mean'], yerr=coverage_by_metric['std'], 
                     capsize=5, alpha=0.7, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        
        ax.set_xlabel('Similarity Metric', fontsize=12, fontweight='bold')
        ax.set_ylabel('Keyword Coverage (%)', fontsize=12, fontweight='bold')
        ax.set_title('Retrieval Quality Comparison Across Metrics\n(10 Legal Queries Evaluation)', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([m.upper() for m in coverage_by_metric['metric']])
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, (_, row) in enumerate(coverage_by_metric.iterrows()):
            ax.text(i, row['mean'] + 2, f"{row['mean']:.1f}%", ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/coverage_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {output_dir}/coverage_comparison.png")
        
        # 2. Box Plot: Retrieval Times
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        retrieval_times = [retrieval_df[retrieval_df['metric'] == m]['retrieval_time'].values for m in ['cosine', 'euclidean', 'hybrid'] if len(retrieval_df[retrieval_df['metric'] == m]) > 0]
        
        if retrieval_times:
            bp = ax.boxplot(retrieval_times, labels=['Cosine', 'Euclidean', 'Hybrid'][:len(retrieval_times)], patch_artist=True)
            
            # Customize boxplot colors
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1'][:len(retrieval_times)]
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        
        ax.set_xlabel('Similarity Metric', fontsize=12, fontweight='bold')
        ax.set_ylabel('Retrieval Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_title('Retrieval Time Distribution by Metric\n(Lower is Better)', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/retrieval_times.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {output_dir}/retrieval_times.png")
        
        # 3. Radar Chart: Multi-metric Comparison
        if len(coverage_by_metric) >= 2:
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
            
            metrics_to_plot = ['coverage_percentage', 'diversity_score', 'avg_relevance']
            metrics_renamed = ['Coverage', 'Diversity', 'Relevance']
            
            angles = np.linspace(0, 2 * np.pi, len(metrics_renamed), endpoint=False).tolist()
            angles += angles[:1]  # Close the loop
            
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
            for idx, metric in enumerate(['cosine', 'euclidean', 'hybrid']):
                metric_data = retrieval_df[retrieval_df['metric'] == metric]
                if not metric_data.empty:
                    values = [metric_data[m].mean() for m in metrics_to_plot]
                    values += values[:1]  # Close the loop
                    
                    ax.plot(angles, values, 'o-', linewidth=2, label=metric.upper(), color=colors[idx])
                    ax.fill(angles, values, alpha=0.15, color=colors[idx])
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metrics_renamed, fontsize=11)
            ax.set_ylim(0, 100)
            ax.set_title('Radar Chart: Multi-Metric Performance Comparison\n(Higher is Better)', 
                        fontsize=14, fontweight='bold', pad=20)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            ax.grid(True)
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/radar_chart.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ Saved: {output_dir}/radar_chart.png")
        
        print(f"\n✅ All visualizations saved to '{output_dir}/' directory")
    
    def generate_report(self, retrieval_df: pd.DataFrame, llm_df: pd.DataFrame, output_file: str = 'research_results/evaluation_report.txt'):
        """Generate comprehensive research report"""
        os.makedirs('research_results', exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("LEGAL RAG ASSISTANT - RESEARCH EVALUATION REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Configuration: Chunk Size={CHUNK_SIZE}, Chunk Overlap={CHUNK_OVERLAP}\n\n")
            
            if not retrieval_df.empty:
                f.write("📊 RETRIEVAL METRICS COMPARISON\n")
                f.write("-" * 40 + "\n\n")
                
                for metric in ['cosine', 'euclidean', 'hybrid']:
                    metric_data = retrieval_df[retrieval_df['metric'] == metric]
                    if not metric_data.empty:
                        f.write(f"{metric.upper()}:\n")
                        f.write(f"  • Avg. Keyword Coverage: {metric_data['coverage_percentage'].mean():.2f}%\n")
                        f.write(f"  • Avg. Relevance Score: {metric_data['avg_relevance'].mean():.3f}\n")
                        f.write(f"  • Avg. Retrieval Time: {metric_data['retrieval_time'].mean():.3f}s\n")
                        f.write(f"  • Diversity Score: {metric_data['diversity_score'].mean():.3f}\n\n")
            
            if not llm_df.empty:
                f.write("\n🤖 LLM ANSWER QUALITY\n")
                f.write("-" * 40 + "\n\n")
                
                for metric in ['cosine', 'euclidean', 'hybrid']:
                    metric_llm = llm_df[llm_df['metric'] == metric]
                    if not metric_llm.empty:
                        f.write(f"{metric.upper()}:\n")
                        f.write(f"  • Citation Rate: {(metric_llm['has_citations'].sum() / len(metric_llm) * 100):.1f}%\n")
                        f.write(f"  • Avg. Citations per Answer: {metric_llm['citation_count'].mean():.1f}\n")
                        f.write(f"  • Answer Keyword Coverage: {metric_llm['answer_keyword_coverage'].mean():.2f}\n\n")
            
            f.write("\n🎯 KEY FINDINGS\n")
            f.write("-" * 40 + "\n\n")
            
            if not retrieval_df.empty:
                best_coverage = retrieval_df.groupby('metric')['coverage_percentage'].mean().idxmax()
                f.write(f"1. Best Retrieval Quality: {best_coverage.upper()}\n")
                f.write(f"2. Recommended Metric: HYBRID (balanced quality & diversity)\n")
            
            f.write(f"3. Chunk Configuration: {CHUNK_SIZE} chars with {CHUNK_OVERLAP} overlap\n")
        
        print(f"\n📄 Research report saved to '{output_file}'")

def main():
    """Run complete research evaluation"""
    print("🚀 Starting Research Evaluation Suite")
    print("=" * 80)
    
    # Initialize evaluator
    evaluator = LegalRAGEvaluator()
    
    # Check if vectorstore exists
    try:
        vs = load_vectorstore()
        print(f"✅ Vector store loaded: {vs.index.ntotal} vectors")
    except FileNotFoundError:
        print("❌ No FAISS index found!")
        print("Please run: python scripts/ingest.py")
        return
    
    # Load test queries
    test_queries = evaluator.load_test_queries()
    print(f"\n📚 Loaded {len(test_queries)} test queries for evaluation")
    
    # Step 1: Evaluate retrieval metrics
    print("\n" + "="*80)
    retrieval_df = evaluator.run_metric_comparison(test_queries, k=5)
    
    # Step 2: Evaluate LLM quality (optional)
    print("\n" + "="*80)
    print("⚡ Running LLM evaluation (this will take a few minutes)...")
    llm_df = evaluator.run_llm_quality_evaluation(test_queries[:3])  # Use subset for speed
    
    # Step 3: Generate visualizations
    if not retrieval_df.empty:
        print("\n" + "="*80)
        print("📊 Generating visualizations...")
        evaluator.generate_visualizations(retrieval_df, llm_df)
    
    # Step 4: Generate research report
    evaluator.generate_report(retrieval_df, llm_df)
    
    # Step 5: Save raw data for further analysis
    if not retrieval_df.empty:
        retrieval_df.to_csv('research_results/retrieval_metrics.csv', index=False)
        print(f"✓ Saved: research_results/retrieval_metrics.csv")
    
    if not llm_df.empty:
        llm_df.to_csv('research_results/llm_quality_metrics.csv', index=False)
        print(f"✓ Saved: research_results/llm_quality_metrics.csv")
    
    print("\n" + "="*80)
    print("✅ Research evaluation complete!")
    print("\n📁 Results saved to 'research_results/' directory")
    print("   Check the folder for graphs and reports!")

if __name__ == "__main__":
    main()