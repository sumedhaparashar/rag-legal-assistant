"""Add this file - for comparing different retrieval methods"""
import time
import json
from typing import List, Dict

class RetrievalEvaluator:
    def __init__(self):
        self.results = {
            'cosine': [],
            'euclidean': [],
            'hybrid': []
        }
    
    def evaluate_retrieval(self, question: str, expected_pages: List[int], 
                          vectorstore, k=5):
        """Compare all three metrics on the same question"""
        metrics_results = {}
        
        # Test Euclidean
        start = time.time()
        euclidean_docs = vectorstore.similarity_search(question, k)
        euclidean_time = time.time() - start
        
        # Test Cosine  
        start = time.time()
        cosine_docs = vectorstore.similarity_search_cosine(question, k)
        cosine_time = time.time() - start
        
        # Test Hybrid
        start = time.time()
        hybrid_docs = vectorstore.hybrid_search(question, k)
        hybrid_time = time.time() - start
        
        return {
            'euclidean': {'docs': euclidean_docs, 'time': euclidean_time},
            'cosine': {'docs': cosine_docs, 'time': cosine_time},
            'hybrid': {'docs': hybrid_docs, 'time': hybrid_time}
        }
    
    def save_results(self, filename='research_results/comparison.json'):
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)