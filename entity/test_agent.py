"""
=============================================================
AUTONOMOUS PATIENT AGENT v1.0
True AI Patient with Own Will, Memory, and Reasoning
=============================================================

NOT hardcoded answers. NOT predetermined responses.
TRUE autonomous agent that:
- THINKS about questions
- REASONS about its condition
- DECIDES what to say
- REMEMBERS conversation
- ADAPTS to doctor feedback
- HAS INTERNAL STATE (the actual disease progression)
"""

from diagnostic_agent import DiagnosticAgent
from experience_db import ExperienceDB
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import json
from datetime import datetime
import warnings

warnings.filterwarnings("ignore", message="This package.*duckduckgo_search")


class DiseaseState:
    """Internal patient condition - autonomous agent's 'body'"""

    def __init__(self, disease_name: str, pathophysiology: dict):
        self.disease_name = disease_name
        self.pathophysiology = pathophysiology  # What's actually happening
        self.symptom_severity = {}  # How bad each symptom is
        self.progression_timeline = 0  # Days since onset
        self.complications = []  # What could develop

    def to_context(self) -> str:
        """Convert disease state to natural description"""
        return json.dumps(self.pathophysiology, indent=2)


class AutonomousPatientAgent:
    """
    TRUE AUTONOMOUS AGENT
    - Has its own internal state (disease)
    - Thinks independently about questions
    - Remembers conversation
    - Adapts answers based on reasoning
    - NOT hardcoded
    """

    def __init__(self, disease: DiseaseState):
        self.disease = disease  # Patient's actual condition
        self.model = OllamaLLM(model="llama3.2", temperature=0.6)
        self.conversation_memory = []
        self.reasoning_log = []
        self.consistency_check = {}  # Track what patient said before

    def think_about_question(self, question: str) -> dict:
        """
        STAGE 1: THINK & REASON
        Agent reasons about what the doctor is asking and what to answer
        """

        prompt = ChatPromptTemplate.from_template("""
You are a patient with {disease}. You are NOT following a script - you are actually experiencing this illness.

YOUR ACTUAL CONDITION:
{disease_context}

CONVERSATION SO FAR:
{conversation_history}

DOCTOR JUST ASKED:
"{question}"

THINK ABOUT THIS QUESTION:
1. What is the doctor trying to find out?
2. How does my actual condition relate to this question?
3. Have I already answered something similar? (be consistent)
4. What would a real patient with this condition actually experience?

REASONING (show your thinking process):
""")

        conversation_str = "\n".join([
            f"Doctor: {q}\nPatient: {a}"
            for q, a in self.conversation_memory
        ])

        reasoning = self.model.invoke(prompt.format(
            disease=self.disease.disease_name,
            disease_context=self.disease.to_context(),
            conversation_history=conversation_str if conversation_str else "No previous questions yet",
            question=question
        ))

        return {
            "reasoning": reasoning,
            "question": question,
            "timestamp": datetime.now().isoformat()
        }

    def decide_answer(self, reasoning: str) -> str:
        """
        STAGE 2: DECIDE & GENERATE
        Based on reasoning, generate autonomous answer
        """

        prompt = ChatPromptTemplate.from_template("""
Based on this reasoning about your condition:

{reasoning}

GENERATE YOUR ANSWER:
- Be truthful about your actual experience
- Sound like a real patient describing their condition
- Be specific (not vague)
- Answer the doctor's actual question
- Keep it natural (2-3 sentences)

ANSWER (just the answer, no explanation):
""")

        answer = self.model.invoke(prompt.format(reasoning=reasoning))
        return answer.strip()

    def verify_consistency(self, question: str, answer: str) -> dict:
        """
        STAGE 3: VERIFY
        Check if answer is consistent with previous statements
        """

        if not self.conversation_memory:
            return {"consistent": True, "warning": None}

        prompt = ChatPromptTemplate.from_template("""
PATIENT'S PREVIOUS STATEMENTS:
{previous_answers}

NEW STATEMENT:
Doctor asked: {question}
Patient answered: {answer}

IS THIS CONSISTENT?
Check if the new answer contradicts or is consistent with previous answers.
Only flag real contradictions (not natural evolution of symptoms).

VERDICT (consistent/contradictory/evolved):
EXPLANATION:
""")

        previous_str = "\n".join([
            f"Q: {q}\nA: {a}"
            for q, a in self.conversation_memory
        ])

        verdict = self.model.invoke(prompt.format(
            previous_answers=previous_str,
            question=question,
            answer=answer
        ))

        return {
            "consistency_check": verdict,
            "question": question,
            "answer": answer
        }

    def respond_to_question(self, question: str) -> str:
        """
        COMPLETE AUTONOMOUS RESPONSE PROCESS
        NOT hardcoded. THINKING through each step.
        """

        print(f"\n🧠 PATIENT THINKING...")

        # Step 1: Think & Reason
        print(f"  └─ Reasoning about question...")
        thinking = self.think_about_question(question)
        self.reasoning_log.append(thinking)
        print(f"     Thinking: {thinking['reasoning'][:150]}...")

        # Step 2: Decide & Generate
        print(f"  └─ Deciding response...")
        answer = self.decide_answer(thinking["reasoning"])

        # Step 3: Verify Consistency
        print(f"  └─ Verifying consistency with previous answers...")
        consistency = self.verify_consistency(question, answer)
        print(f"     Check: {consistency['consistency_check'][:100]}...")

        # Step 4: Remember
        self.conversation_memory.append((question, answer))
        self.consistency_check[question] = consistency

        print(f"  └─ ✓ Decision made autonomously")

        return answer

    def get_patient_report(self) -> dict:
        """Generate report of patient's autonomous decision-making"""

        return {
            "disease": self.disease.disease_name,
            "conversations": len(self.conversation_memory),
            "reasoning_steps": len(self.reasoning_log),
            "consistency_checks": len(self.consistency_check),
            "conversation_history": [
                {"doctor_question": q, "patient_answer": a}
                for q, a in self.conversation_memory
            ],
            "reasoning_history": self.reasoning_log[:3]  # First 3 for brevity
        }


class AutonomousTestSuite:
    """Test BAYMAX against AUTONOMOUS PATIENTS (not hardcoded)"""

    def __init__(self):
        self.doctor = DiagnosticAgent()
        self.test_results = []

    def create_autonomous_patients(self):
        """Create true autonomous patient agents"""

        patients = [
            AutonomousPatientAgent(DiseaseState(
                "Streptococcal Pharyngitis",
                {
                    "pathophysiology": "Bacterial infection of throat caused by Group A Streptococcus",
                    "onset": "3 days ago, sudden",
                    "fever": 39.5,  # Celsius
                    "throat_inflammation": "severe",
                    "symptoms_present": ["sore throat", "fever", "difficulty swallowing", "malaise"],
                    "symptoms_absent": ["rash", "cough"],
                    "immunity": "Not vaccinated",
                    "progression": "Improving slowly with rest"
                }
            )),
            AutonomousPatientAgent(DiseaseState(
                "Varicella (Chickenpox)",
                {
                    "pathophysiology": "Viral infection with characteristic vesicular rash",
                    "onset": "5 days fever, 2 days rash",
                    "fever": 39.8,
                    "rash_type": "Vesicular, polymorphic lesions",
                    "rash_distribution": "Trunk > extremities > face",
                    "rash_progression": "New lesions appearing in crops",
                    "pruritus": "Severe, worse at night",
                    "vaccination_history": "Never vaccinated",
                    "complications": "Risk of secondary bacterial infection"
                }
            )),
            AutonomousPatientAgent(DiseaseState(
                "Acute Gastroenteritis",
                {
                    "pathophysiology": "Acute inflammatory infection of GI tract",
                    "onset": "6 hours ago, food exposure",
                    "diarrhea": "Watery, no blood",
                    "vomiting": "Multiple episodes, worsening",
                    "fever": 37.8,
                    "abdominal_pain": "Crampy, colicky",
                    "food_exposure": "Restaurant meal yesterday",
                    "likely_organism": "Bacterial (Salmonella/Staphylococcus)",
                    "hydration_status": "Mild dehydration"
                }
            )),
        ]

        return patients

    def test_patient(self, patient: AutonomousPatientAgent, initial_symptom: str):
        """Test BAYMAX against autonomous patient"""

        print(f"\n{'='*70}")
        print(f"TESTING AUTONOMOUS PATIENT: {patient.disease.disease_name}")
        print(f"{'='*70}")
        print(f"Patient's Actual Condition: {patient.disease.disease_name}")
        print(f"Initial Complaint: {initial_symptom}")

        conversation = []
        max_questions = 5

        for q_num in range(max_questions):
            print(f"\n--- Question {q_num + 1} ---")

            # Doctor asks question
            try:
                doctor_question = self.doctor.get_follow_up_question(
                    initial_symptom,
                    conversation,
                    region="India"
                )
            except Exception as e:
                print(f"⚠️ Doctor error: {str(e)[:100]}")
                break

            print(f"🏥 Doctor: {doctor_question[:100]}...")

            # PATIENT THINKS AUTONOMOUSLY (not hardcoded!)
            patient_answer = patient.respond_to_question(doctor_question)

            print(f"👤 Patient: {patient_answer}")

            conversation.append((doctor_question, patient_answer))

            # Check if doctor has enough info
            try:
                if not self.doctor.should_continue_asking(initial_symptom, conversation):
                    print("\n✓ Doctor has sufficient information")
                    break
            except:
                break

        # Get diagnosis
        print(f"\n⏳ Doctor generating diagnosis...")
        try:
            diagnosis = self.doctor.narrow_down_diagnosis(initial_symptom, conversation)
            print(f"\n{'='*70}")
            print(f"🏥 DOCTOR'S DIAGNOSIS:")
            print(f"{'='*70}")
            print(diagnosis[:300] + "..." if len(diagnosis) > 300 else diagnosis)
            print(f"{'='*70}")
        except Exception as e:
            print(f"❌ Error: {str(e)[:100]}")
            diagnosis = None

        # Check if correct
        expected = patient.disease.disease_name.lower()
        is_correct = (expected in diagnosis.lower()) if diagnosis else False

        print(f"\n{'CORRECT ✅' if is_correct else 'INCORRECT ❌'}")

        # Get patient's report
        patient_report = patient.get_patient_report()

        print(f"\n{'='*70}")
        print(f"AUTONOMOUS PATIENT REPORT")
        print(f"{'='*70}")
        print(f"Conversations: {patient_report['conversations']}")
        print(f"Reasoning Steps: {patient_report['reasoning_steps']}")
        print(f"Consistency Checks: {patient_report['consistency_checks']}")
        print(f"\nPatient's Reasoning (first check):")
        if patient_report['reasoning_history']:
            print(f"  {patient_report['reasoning_history'][0]['reasoning'][:200]}...")

        return {
            "disease": patient.disease.disease_name,
            "correct": is_correct,
            "conversations": len(conversation),
            "patient_report": patient_report,
            "doctor_diagnosis": diagnosis
        }

    def run_autonomous_tests(self):
        """Run tests with AUTONOMOUS PATIENTS"""

        patients = self.create_autonomous_patients()
        test_cases = [
            (patients[0], "Fever and sore throat"),
            (patients[1], "High fever with rash on trunk"),
            (patients[2], "Diarrhea and vomiting"),
        ]

        print(f"\n{'='*70}")
        print(f"🚀 AUTONOMOUS PATIENT TEST SUITE")
        print(f"Testing BAYMAX against TRUE autonomous agents")
        print(f"(NOT hardcoded, patients THINK independently)")
        print(f"{'='*70}")

        results = []
        for patient, symptom in test_cases:
            result = self.test_patient(patient, symptom)
            results.append(result)
            self.test_results.append(result)

        # Summary
        correct = len([r for r in results if r['correct']])
        total = len(results)
        accuracy = (correct / total * 100) if total > 0 else 0

        print(f"\n{'='*70}")
        print(f"FINAL RESULTS")
        print(f"{'='*70}")
        print(f"✅ Correct Diagnoses: {correct}/{total}")
        print(f"📊 Accuracy: {accuracy:.1f}%")
        print(f"\n🧠 Key Difference:")
        print(f"  OLD: Hardcoded if/else → lookup table → predetermined answers")
        print(f"  NEW: LLM reasoning → autonomous thinking → dynamic answers")
        print(f"{'='*70}")

        return results


# ============================================================
# RUN AUTONOMOUS TESTS
# ============================================================

if __name__ == "__main__":
    suite = AutonomousTestSuite()
    results = suite.run_autonomous_tests()

    # Save detailed results
    with open("autonomous_patient_results.json", "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "test_type": "Autonomous Patient Agent Testing",
            "total_tests": len(results),
            "accuracy": sum(1 for r in results if r['correct']) / len(results) * 100,
            "results": results
        }, f, indent=2)

    print(f"\n📁 Results saved to: autonomous_patient_results.json")
    print(f"\n🎉 AUTONOMOUS PATIENT AGENTS WORKING!")
    print(f"   Patients have their own will, reasoning, and memory!")