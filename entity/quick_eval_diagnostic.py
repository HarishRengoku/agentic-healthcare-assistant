"""
QUICK EVALUATION - DIAGNOSTIC AGENT ONLY
Satisfies: NLP Metrics (Similarity) & Classification Metrics (Accuracy)
"""

import time
import numpy as np
from diagnostic_agent import DiagnosticAgent

# --- 1. SETUP METRICS (SIMPLIFIED) ---
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    print("✅ Loading embedding model for Similarity Metric (BERTScore proxy)...")
    # Small, fast model for semantic similarity
    model = SentenceTransformer('all-MiniLM-L6-v2')
    HAS_NLP = True
except ImportError:
    print("⚠️  'sentence-transformers' not found. Using basic text overlap for Similarity.")
    HAS_NLP = False

def calculate_similarity(text1, text2):
    """Calculates Cosine Similarity between two texts"""
    if HAS_NLP:
        # Vectorize and calculate cosine similarity
        embeddings = model.encode([text1, text2])
        return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    else:
        # Fallback: Simple word overlap (Jaccard index)
        set1 = set(text1.lower().split())
        set2 = set(text2.lower().split())
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0

# --- 2. DEFINE TEST DATA (Ground Truth) ---
# Format: (Initial Symptom, Ideal Doctor Question, Actual Disease)
TEST_CASES = [
    {
        "symptoms": "High fever, dry cough, chest pain",
        "ideal_question": "When did the fever start and do you have shortness of breath?",
        "ground_truth_diagnosis": "Pneumonia"
    },
    {
        "symptoms": "Severe headache, stiff neck, sensitivity to light",
        "ideal_question": "When did this start Have you noticed any rash or confusion?",
        "ground_truth_diagnosis": "Meningitis"
    },
    {
        "symptoms": "Watery diarrhea, vomiting, stomach cramps",
        "ideal_question": "When did this start Have you eaten any street food or travelled recently?",
        "ground_truth_diagnosis": "Gastroenteritis"
    },
    {
        "symptoms": "Sudden chest pain radiating to left arm",
        "ideal_question": "When did this start Do you have a history of heart disease or high blood pressure?",
        "ground_truth_diagnosis": "Myocardial Infarction"
    },
    {
        "symptoms": "Itchy red rash on face and swelling",
        "ideal_question": "When did this start Did you eat anything new or take new medication?",
        "ground_truth_diagnosis": "Allergic Reaction"
    }
]

# --- 3. RUN EVALUATION ---
def run_quick_eval():
    print("\n" + "="*60)
    print("🚀 STARTING QUICK DIAGNOSTIC AGENT EVALUATION")
    print("="*60)
    
    try:
        agent = DiagnosticAgent()
        print("✅ Diagnostic Agent loaded successfully.\n")
    except Exception as e:
        print(f"❌ Error loading agent: {e}")
        return

    similarity_scores = []
    accuracies = []
    latencies = []

    for i, case in enumerate(TEST_CASES, 1):
        print(f"Processing Case {i}: {case['ground_truth_diagnosis']}...")
        
        # A. Measure Question Generation (NLP Metric)
        start_time = time.time()
        
        # Ask agent to generate a question
        generated_question = agent.get_follow_up_question(
            case['symptoms'], 
            [] # No history yet
        )
        
        latency = time.time() - start_time
        latencies.append(latency)
        
        # Calculate Similarity (Generated vs Ideal)
        sim_score = calculate_similarity(generated_question, case['ideal_question'])
        similarity_scores.append(sim_score)
        
        print(f"  - Symptoms: {case['symptoms']}")
        print(f"  - Agent Asked: \"{generated_question}\"")
        print(f"  - Ideal Q:     \"{case['ideal_question']}\"")
        print(f"  - Similarity Score: {sim_score:.4f}")

        # B. Measure Diagnosis Accuracy (Classification Metric)
        # We simulate a conversation to get a diagnosis
        # (For speed, we just feed the ground truth symptoms directly to narrow_down)
        conversation = [("What are your symptoms?", case['symptoms'])]
        
        diagnosis_output = agent.narrow_down_diagnosis(case['symptoms'], conversation)
        
        # Check if ground truth is in the diagnosis text
        is_correct = case['ground_truth_diagnosis'].lower() in diagnosis_output.lower()
        accuracies.append(1 if is_correct else 0)
        
        result_icon = "✅" if is_correct else "❌"
        print(f"  - Diagnosis Result: {result_icon} (Expected: {case['ground_truth_diagnosis']})")
        print("-" * 60)

    # --- 4. PRINT FINAL REPORT ---
    avg_similarity = np.mean(similarity_scores)
    avg_accuracy = np.mean(accuracies)
    avg_latency = np.mean(latencies)

    print("\n" + "="*60)
    print("📊 FINAL EVALUATION REPORT FOR DR. THENMOZHI")
    print("="*60)
    print(f"Component: Diagnostic Agent (LLM)")
    print("-" * 60)
    
    print(f"1. NLP METRIC (Question Relevance)")
    print(f"   Metric Used: Cosine Similarity (Semantic)")
    print(f"   Score: {avg_similarity:.4f} / 1.0")
    print(f"   Verdict: {'Excellent' if avg_similarity > 0.5 else 'Average'}")
    
    print(f"\n2. CLASSIFICATION METRIC (Diagnostic Success)")
    print(f"   Metric Used: Accuracy")
    print(f"   Score: {avg_accuracy:.2%} ({sum(accuracies)}/{len(accuracies)} correct)")
    
    print(f"\n3. PERFORMANCE METRIC")
    print(f"   Metric Used: Latency per Query")
    print(f"   Average Time: {avg_latency:.2f} seconds")
    print("="*60)
    print("Evaluation Complete. Copy these numbers to your report.")

if __name__ == "__main__":
    run_quick_eval()
