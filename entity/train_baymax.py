"""
🏥 BAYMAX TRAINING MONTAGE SIMULATOR (FULL DIAGNOSIS LOGGING)
Generates 100 synthetic patients to train the Experience Database.
Shows ENTIRE conversation + FULL DIAGNOSIS!
"""

import time
import random
import json
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from diagnostic_agent import DiagnosticAgent

# --- CONFIGURATION ---
NUM_CASES_TO_GENERATE = 100
MODEL_NAME = "llama3.2"

# --- PATIENT PERSONA GENERATOR ---
class PatientSimulator:
    def __init__(self):
        self.model = OllamaLLM(model=MODEL_NAME, temperature=0.7)

    def generate_patient_profile(self, condition, variation):
        """Creates a unique patient persona"""
        prompt = f"""
        Generate a brief patient profile for someone with {condition}.
        Variation: {variation}
        
        Output format (JSON):
        {{
            "initial_symptom": "one sentence describing main complaint",
            "history": "brief history (onset, severity, triggers)",
            "hidden_info": "key diagnostic clue to reveal only if asked"
        }}
        """
        try:
            res = self.model.invoke(prompt)
            try:
                start = res.find('{')
                end = res.rfind('}') + 1
                data = json.loads(res[start:end])
                return data
            except:
                return {
                    "initial_symptom": f"I have symptoms of {condition}",
                    "history": "It started recently.",
                    "hidden_info": "No specific details."
                }
        except Exception as e:
            print(f"Error generating profile: {e}")
            return None

    def answer_question(self, profile, doctor_question, conversation_history):
        """Simulates patient answering doctor"""
        history_text = "\n".join([f"Dr: {q}\nMe: {a}" for q, a in conversation_history])

        prompt = f"""
        You are a patient.
        Your Condition: {profile['initial_symptom']}
        History: {profile['history']}
        Hidden Info: {profile['hidden_info']}
        
        Conversation so far:
        {history_text}
        
        Doctor's Question: "{doctor_question}"
        
        Task: Answer the doctor naturally based ONLY on your profile. 
        Keep it brief (1-2 sentences). Don't reveal the diagnosis name, just symptoms.
        """
        return self.model.invoke(prompt).strip()

# --- TRAINING RUNNER ---
def run_training_montage():
    print("="*80)
    print("🥊 BAYMAX TRAINING MONTAGE STARTING...")
    print(f"Goal: {NUM_CASES_TO_GENERATE} Cases with FULL Conversation + FULL Diagnosis")
    print("="*80)

    agent = DiagnosticAgent()
    patient_sim = PatientSimulator()

    # diverse conditions to learn
    conditions = [
        "Migraine", "Flu", "Gastroenteritis", "Pneumonia", "Anemia",
        "Anxiety", "Dermatitis", "Asthma", "Hypertension", "Diabetes (Type 2)",
        "Food Poisoning", "Covid-19", "Concussion", "Allergic Rhinitis", "Acid Reflux",
        "Insomnia", "Carpal Tunnel", "Kidney Stone", "Sunburn", "Dehydration"
    ]

    variations = ["Young Adult", "Elderly", "Child (parent speaking)", "Athlete", "Office Worker"]

    cases_completed = 0

    for i in range(NUM_CASES_TO_GENERATE):
        # 1. Setup Case
        condition = conditions[i % len(conditions)]
        variation = variations[i % len(variations)]

        print(f"\n{'='*80}")
        print(f"[CASE {i+1}/{NUM_CASES_TO_GENERATE}] {condition} ({variation})")
        print('='*80)

        profile = patient_sim.generate_patient_profile(condition, variation)
        if not profile:
            print(f"❌ Failed to generate profile for {condition}. Skipping...")
            continue

        print(f"👤 PATIENT PROFILE:")
        print(f"   Initial Complaint: {profile['initial_symptom']}")
        print(f"   History: {profile['history']}")
        print(f"   Hidden Info: {profile['hidden_info']}")

        # 2. Run Diagnostic Loop
        history = []
        initial_symptom = profile['initial_symptom']

        print(f"\n📞 CONVERSATION BEGINS:\n")

        # Ask up to 5 questions (respecting your agent's logic)
        question_count = 0
        try:
            while agent.should_continue_asking(initial_symptom, history) and question_count < 5:
                question_count += 1

                question = agent.get_follow_up_question(initial_symptom, history)
                answer = patient_sim.answer_question(profile, question, history)
                history.append((question, answer))

                # FULL CONVERSATION LOGGING
                print(f"Q{question_count}: {question}")
                print(f"A{question_count}: {answer}")
                print()
        except KeyboardInterrupt:
            print("\n❌ Training interrupted by user (Ctrl+C)")
            break
        except Exception as e:
            print(f"\n⚠️  Error in conversation loop: {str(e)[:100]}")
            print("Continuing to next case...")
            continue

        # 3. Diagnose & Store (GET FULL DIAGNOSIS)
        print(f"🏥 BAYMAX MAKING DIAGNOSIS (After {question_count} questions)...")
        try:
            diagnosis = agent.narrow_down_diagnosis(initial_symptom, history)
        except Exception as e:
            print(f"⚠️  Error getting diagnosis: {str(e)[:100]}")
            diagnosis = f"{condition}: Unable to generate diagnosis due to error."

        # PRINT FULL DIAGNOSIS (NOT TRUNCATED!)
        print(f"\n📋 FULL DIAGNOSIS:\n{diagnosis}\n")

        # EXTRACT CONFIDENCE
        confidence = "High" if "High" in diagnosis else "Medium"

        # *** STORE EXPERIENCE WITH FULL DIAGNOSIS ***
        try:
            agent.experience_db.store_case(
                initial_symptoms=initial_symptom,
                conversation_history=history,
                final_diagnosis=diagnosis,  # ← FULL DIAGNOSIS, NOT TRUNCATED
                confidence=confidence
            )
            cases_completed += 1
            print(f"✅ Case #{agent.experience_db.get_case_count()} stored COMPLETELY in Experience Database")
        except Exception as e:
            print(f"❌ Error storing case: {str(e)[:100]}")

    print("\n" + "="*80)
    print("🏆 TRAINING MONTAGE COMPLETE!")
    print(f"Total Cases Created: {cases_completed}")
    print(f"Total Experience DB Size: {agent.experience_db.get_case_count()}")
    print("BAYMAX is now HIGHLY EXPERIENCED! 🩺")
    print("="*80)

if __name__ == "__main__":
    try:
        run_training_montage()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training stopped by user. Database has been saved with cases completed so far.")
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")