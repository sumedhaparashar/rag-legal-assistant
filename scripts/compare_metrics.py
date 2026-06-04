"""Run this once to compare performance for your paper"""
import sys
sys.path.append('.')
from src.vectorstore import VectorStoreManager
from src.evaluation import RetrievalEvaluator
import json

# Test questions for evaluation
test_questions = [
    "What are the duties of company directors?",
    "How can a company be wound up?",
    "What are the requirements for annual general meetings?",
    # Add 10-15 more questions for your research
]

def main():
    # Load your existing vectorstore
    vstore = VectorStoreManager()
    vstore.load_vectorstore('data/vectorstore')
    
    evaluator = RetrievalEvaluator()
    
    for q in test_questions:
        print(f"Evaluating: {q}")
        # You'll need to define expected pages based on your documents
        results = evaluator.evaluate_retrieval(q, [], vstore)
        evaluator.results['cosine'].append(results['cosine']['time'])
        evaluator.results['euclidean'].append(results['euclidean']['time'])
        evaluator.results['hybrid'].append(results['hybrid']['time'])
    
    # Save results for your paper
    evaluator.save_results()
    print("Results saved to research_results/comparison.json")

if __name__ == "__main__":
    main()