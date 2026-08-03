"""
ADVANCED EVALUATION - TRANSFORMER METRICS
Implements Perplexity, BERTScore, and mAP (Diagnostic Ranking)
"""

import torch
import numpy as np
from diagnostic_agent import DiagnosticAgent

# --- PREREQUISITES ---
# pip install torch transformers bert_score

try:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    from bert_score import score as bert_score
    print("✅ Transformers & BERTScore libraries loaded.")
except ImportError:
    print("❌ MISSING LIBRARIES. Run: pip install torch transformers bert_score")
    exit()

# --- 1. SETUP METRICS ---

class AdvancedMetrics:
    def __init__(self):
        print("⏳ Loading GPT-2 for Perplexity calculation (this measures fluency)...")
        self.ppl_model_id = 'gpt2' # Small model for speed
        self.ppl_tokenizer = GPT2Tokenizer.from_pretrained(self.ppl_model_id)
        self.ppl_model = GPT2LMHeadModel.from_pretrained(self.ppl_model_id)
        self.ppl_model.eval()

    def calculate_perplexity(self, text):
        """
        Calculates fluency perplexity using GPT-2.
        Lower is better (means text is natural/fluent).
        """
        encodings = self.ppl_tokenizer(text, return_tensors='pt')
        max_length = self.ppl_model.config.n_positions
        stride = 512
        seq_len = encodings.input_ids.size(1)

        nlls = []
        prev_end_loc = 0
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc
            input_ids = encodings.input_ids[:, begin_loc:end_loc]
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = self.ppl_model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss

            nlls.append(neg_log_likelihood)
            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        ppl = torch.exp(torch.stack(nlls).mean())
        return ppl.item()

    def calculate_bertscore(self, candidate, reference):
        """
        Calculates BERTScore (Similarity).
        F1 is the main metric. Range 0 to 1.
        """
        # We use a small model for speed, e.g., distilbert-base-uncased
        # hash_code ensures reproducibility
        P, R, F1 = bert_score([candidate], [reference], lang='en', verbose=False)
        return F1.mean().item()

    def calculate_map_diagnosis(self, predicted_text, correct_diagnosis):
        """
        Adapts mAP for Diagnosis.
        We treat the doctor's output as a 'ranked list' of text.
        If the correct diagnosis appears early, score is higher.
        """
        # Simple heuristic:
        # If diagnosis is in first sentence -> Score 1.0
        # If diagnosis is in text but later -> Score 0.5
        # If not found -> Score 0.0
        
        pred_lower = predicted_text.lower()
        target_lower = correct_diagnosis.lower()
        
        if target_lower not in pred_lower:
            return 0.0
        
        # Check if it's in the first 100 chars (approx first sentence/header)
        if target_lower in pred_lower[:100]:
            return 1.0
        
        return 0.5

# --- 2. TEST DATA ---
TEST_CASES = [
    {
        "symptoms": "High fever, dry cough, chest pain",
        "ideal_response": "When did the fever start and do you have shortness of breath?",
        "diagnosis": "Pneumonia"
    },
    {
        "symptoms": "Severe headache, stiff neck, sensitivity to light",
        "ideal_response": "Have you noticed any rash or confusion? This could be meningitis.", 
        "diagnosis": "Meningitis"
    }
]

# --- 3. RUN EVALUATION ---
def run_eval():
    print("\n" + "="*60)
    print("🧪 STARTING TRANSFORMER METRICS EVALUATION")
    print("="*60)
    
    metrics = AdvancedMetrics()
    agent = DiagnosticAgent()
    
    ppl_scores = []
    bert_scores = []
    map_scores = []

    for i, case in enumerate(TEST_CASES, 1):
        print(f"\nProcessing Case {i}: {case['diagnosis']}")
        
        # 1. Generate Question
        generated_question = agent.get_follow_up_question(case['symptoms'], [])
        print(f"  Agent Asked: \"{generated_question}\"")
        
        # 2. Calculate Perplexity (Fluency)
        ppl = metrics.calculate_perplexity(generated_question)
        ppl_scores.append(ppl)
        print(f"  ✓ Perplexity: {ppl:.2f} (Lower is better)")
        
        # 3. Calculate BERTScore (Similarity to Ideal)
        bs = metrics.calculate_bertscore(generated_question, case['ideal_response'])
        bert_scores.append(bs)
        print(f"  ✓ BERTScore F1: {bs:.4f} (Higher is better)")
        
        # 4. Generate Diagnosis for mAP
        # Simulate conversation for diagnosis
        convo = [("What are symptoms?", case['symptoms'])]
        diagnosis_text = agent.narrow_down_diagnosis(case['symptoms'], convo)
        
        # 5. Calculate mAP (Ranking Accuracy)
        map_val = metrics.calculate_map_diagnosis(diagnosis_text, case['diagnosis'])
        map_scores.append(map_val)
        print(f"  ✓ mAP Score: {map_val:.2f} (Rank Correctness)")

    # --- REPORT ---
    print("\n" + "="*60)
    print("📊 TRANSFORMER METRICS REPORT")
    print("="*60)
    print(f"1. PERPLEXITY (Fluency): {np.mean(ppl_scores):.2f}")
    print(f"   (Standard GPT-2 PPL. <50 is good, <20 is excellent)")
    
    print(f"\n2. BERTSCORE (Semantic Match): {np.mean(bert_scores):.4f}")
    print(f"   (Compare to ideal doctor questions. >0.85 is great)")
    
    print(f"\n3. mAP (Diagnostic Ranking): {np.mean(map_scores):.2f}")
    print(f"   (Did the correct diagnosis appear prominently?)")
    print("="*60)

if __name__ == "__main__":
    run_eval()
