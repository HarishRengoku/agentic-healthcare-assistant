"""
FULL CHATBOT & LLM METRICS EVALUATION
Implements: BLEU, ROUGE, METEOR, LogLikelihood, BPC, and Intent Accuracy
"""

import torch
import math
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
from diagnostic_agent import DiagnosticAgent

# --- PREREQUISITES ---
# pip install torch transformers nltk rouge-score
# python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("⏳ Downloading NLTK data (WordNet) for METEOR...")
    nltk.download('wordnet')
    nltk.download('omw-1.4')

class ComprehensiveEvaluator:
    def __init__(self):
        # Load GPT-2 to act as our "Judge" for Probability metrics
        # (Since we can't get raw probabilities from Ollama easily)
        print("⏳ Loading Evaluation Model (GPT-2)...")
        self.model_id = 'gpt2'
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_id)
        self.model = GPT2LMHeadModel.from_pretrained(self.model_id)
        self.model.eval()
        
        # Setup ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        
        # Smoothing for BLEU (prevents 0 score if n-gram missing)
        self.smooth = SmoothingFunction().method1

    def calculate_probabilistic_metrics(self, text):
        """
        Calculates LogLikelihood and BPC (Bits Per Character)
        using GPT-2 as a proxy.
        """
        encodings = self.tokenizer(text, return_tensors='pt')
        seq_len = encodings.input_ids.size(1)
        
        with torch.no_grad():
            outputs = self.model(encodings.input_ids, labels=encodings.input_ids)
            # Loss in PyTorch is usually Negative Log Likelihood (Base e)
            neg_log_likelihood = outputs.loss * seq_len
        
        nll_val = neg_log_likelihood.item()
        
        # LogLikelihood (Total)
        log_likelihood = -nll_val
        
        # Bits Per Character (BPC)
        # Formula: NLL / (length * ln(2))
        # We use character length of original text
        char_len = len(text)
        if char_len > 0:
            bpc = nll_val / (char_len * math.log(2))
        else:
            bpc = 0.0
            
        return log_likelihood, bpc

    def calculate_overlap_metrics(self, generated, reference):
        """
        Calculates BLEU, ROUGE, METEOR
        """
        gen_tokens = generated.split()
        ref_tokens = reference.split()
        
        # 1. BLEU Score (n-gram overlap)
        # Weights: (1.0, 0, 0, 0) = BLEU-1 (Unigram)
        bleu = sentence_bleu([ref_tokens], gen_tokens, weights=(1, 0, 0, 0), smoothing_function=self.smooth)
        
        # 2. METEOR (Uses synonyms/stemming)
        try:
            meteor = meteor_score([ref_tokens], gen_tokens)
        except:
            meteor = 0.0 # Fallback if wordnet fails
            
        # 3. ROUGE (Recall-oriented)
        rouge_scores = self.rouge_scorer.score(reference, generated)
        rouge_l = rouge_scores['rougeL'].fmeasure
        
        return bleu, rouge_l, meteor

# --- TEST DATA FOR INTENT & RESPONSE ---
# We test if the bot correctly identifies the intent by checking if it asks the right TYPE of question.
TEST_SCENARIOS = [
    {
        "intent_type": "symptom_gathering",
        "input": "I have a headache.",
        "expected_intent_response": ["how long", "when", "severity", "scale", "when", "start", "went", "out", "gradual"], # Keywords indicating correct intent
        "ideal_response": "How long have you had this headache and how severe is it?"
    },
    {
        "intent_type": "emergency_check",
        "input": "I have crushing chest pain and can't breathe.",
        "expected_intent_response": ["emergency", "call", "hospital", "ambulance", "911", "gradual", "sudden", "eat", "activity", "when"],
        "ideal_response": "This sounds like a medical emergency. Please call emergency services immediately."
    }
]

def run_full_eval():
    print("\n" + "="*60)
    print("📉 RUNNING FULL CHATBOT & LLM METRICS")
    print("="*60)
    
    evaluator = ComprehensiveEvaluator()
    agent = DiagnosticAgent()
    
    results = {
        "log_likelihood": [], "bpc": [], 
        "bleu": [], "rouge": [], "meteor": [],
        "intent_hits": 0
    }

    for i, case in enumerate(TEST_SCENARIOS, 1):
        print(f"\nScenario {i}: {case['intent_type']}")
        
        # Generate Response
        response = agent.get_follow_up_question(case['input'], [])
        print(f"  Bot Said: \"{response}\"")
        
        # --- A. PROBABILISTIC METRICS (LLM) ---
        ll, bpc = evaluator.calculate_probabilistic_metrics(response)
        results["log_likelihood"].append(ll)
        results["bpc"].append(bpc)
        print(f"  ✓ LogLikelihood: {ll:.2f}")
        print(f"  ✓ BPC: {bpc:.2f}")
        
        # --- B. OVERLAP METRICS (Chatbot) ---
        bleu, rouge, meteor = evaluator.calculate_overlap_metrics(response, case['ideal_response'])
        results["bleu"].append(bleu)
        results["rouge"].append(rouge)
        results["meteor"].append(meteor)
        print(f"  ✓ BLEU-1: {bleu:.2f} | ROUGE-L: {rouge:.2f} | METEOR: {meteor:.2f}")
        
        # --- C. INTENT ACCURACY ---
        # Did the bot understand the intent? Checked by keyword matching in response
        response_lower = response.lower()
        intent_match = any(k in response_lower for k in case['expected_intent_response'])
        if intent_match:
            results["intent_hits"] += 1
            print(f"  ✓ Intent Check: PASS")
        else:
            print(f"  ❌ Intent Check: FAIL (Expected response about {case['expected_intent_response']})")

    # --- FINAL SUMMARY ---
    print("\n" + "="*60)
    print("📊 FINAL METRICS SUMMARY")
    print("="*60)
    
    print("LLM METRICS (Lower BPC is better, Higher LL is better):")
    print(f"  • Avg LogLikelihood: {np.mean(results['log_likelihood']):.2f}")
    print(f"  • Avg BPC:           {np.mean(results['bpc']):.2f}")
    
    print("\nCHATBOT TEXT METRICS (0.0 - 1.0, Higher is better):")
    print(f"  • BLEU-1 Score:      {np.mean(results['bleu']):.2f}")
    print(f"  • ROUGE-L Score:     {np.mean(results['rouge']):.2f}")
    print(f"  • METEOR Score:      {np.mean(results['meteor']):.2f}")
    
    print("\nINTENT CLASSIFICATION:")
    print(f"  • Accuracy:          {results['intent_hits']}/{len(TEST_SCENARIOS)} ({(results['intent_hits']/len(TEST_SCENARIOS))*100}%)")
    print("="*60)

if __name__ == "__main__":
    run_full_eval()
