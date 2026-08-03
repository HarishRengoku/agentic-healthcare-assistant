"""
BAYMAX Experience Database
Stores past diagnoses and learns from them
"""

import json
import os
from datetime import datetime
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate


class ExperienceDB:
    """Stores and retrieves past diagnostic cases"""

    def __init__(self, db_path="./baymax_experiences"):
        """Initialize the experience database"""
        self.db_path = db_path
        self.experiences_file = os.path.join(db_path, "cases.json")
        self.model = OllamaLLM(model="llama3.2", temperature=0.3)

        # Create directory if it doesn't exist
        os.makedirs(db_path, exist_ok=True)

        # Initialize cases file if empty
        if not os.path.exists(self.experiences_file):
            self._save_cases([])

    def _load_cases(self):
        """Load all past cases from file"""
        try:
            with open(self.experiences_file, 'r') as f:
                return json.load(f)
        except:
            return []

    def _save_cases(self, cases):
        """Save cases to file"""
        with open(self.experiences_file, 'w') as f:
            json.dump(cases, f, indent=2)

    def store_case(self, initial_symptoms, conversation_history, final_diagnosis, confidence):
        """
        Store a completed case in the experience database

        Args:
            initial_symptoms: What patient first reported
            conversation_history: List of (Q, A) tuples
            final_diagnosis: The diagnosis reached
            confidence: Confidence level (High/Medium/Low)
        """

        case = {
            "case_id": len(self._load_cases()) + 1,
            "timestamp": datetime.now().isoformat(),
            "initial_symptoms": initial_symptoms,
            "conversation": [
                {"question": q, "answer": a} for q, a in conversation_history
            ],
            "final_diagnosis": final_diagnosis,
            "confidence": confidence
        }

        cases = self._load_cases()
        cases.append(case)
        self._save_cases(cases)

        print(f"\n Case #{case['case_id']} stored in experience database")

    def find_similar_cases(self, current_symptoms, top_k=3):
        """
        Find similar past cases based on current symptoms

        Args:
            current_symptoms: Patient's current symptoms
            top_k: Number of similar cases to retrieve

        Returns:
            List of similar past cases
        """

        cases = self._load_cases()

        if not cases:
            return []

        # Use LLM to find semantic similarity
        prompt_template = ChatPromptTemplate.from_template("""
You are comparing a current patient's symptoms to past cases BAYMAX has diagnosed.

CURRENT PATIENT SYMPTOMS: {current_symptoms}

PAST CASES:
{past_cases_text}

TASK: Identify which past cases are MOST SIMILAR to the current patient's symptoms.
Return ONLY the case IDs that are similar (e.g., "1, 3, 5" or "none").
Be strict - only return cases with similar symptom patterns.
""")

        # Format past cases for comparison
        past_cases_text = "\n\n".join([
            f"Case #{case['case_id']}: {case['initial_symptoms']} → Diagnosed as {case['final_diagnosis']}"
            for case in cases[-10:]  # Check last 10 cases for performance
        ])

        chain = prompt_template | self.model
        response = chain.invoke({
            "current_symptoms": current_symptoms,
            "past_cases_text": past_cases_text
        })

        # Parse response to get case IDs
        try:
            similar_ids = [int(x.strip()) for x in response.split(",") if x.strip().isdigit()]
            similar_cases = [c for c in cases if c['case_id'] in similar_ids]
            return similar_cases[:top_k]
        except:
            return []

    def get_experience_insight(self, current_symptoms, similar_cases):
        """
        Generate insight from past similar cases

        Args:
            current_symptoms: Current patient symptoms
            similar_cases: Past similar cases

        Returns:
            Clinical insight based on experience
        """

        if not similar_cases:
            return None

        prompt_template = ChatPromptTemplate.from_template("""
You are BAYMAX, a medical diagnostic AI with experience.
You're analyzing a NEW patient and have found SIMILAR PAST CASES in your experience.

NEW PATIENT SYMPTOMS: {current_symptoms}

PAST SIMILAR CASES YOU'VE TREATED:
{similar_cases_text}

TASK: Based on your experience with similar cases, provide a brief clinical insight:
1. What pattern do you notice?
2. What diagnosis was most common in similar cases?
3. What key questions should you prioritize asking?

Format as a short clinical memo (2-3 sentences).
""")

        similar_cases_text = "\n\n".join([
            f"Case #{case['case_id']} ({case['timestamp'][:10]}): {case['initial_symptoms']} → {case['final_diagnosis']} (Confidence: {case['confidence']})"
            for case in similar_cases
        ])

        chain = prompt_template | self.model
        insight = chain.invoke({
            "current_symptoms": current_symptoms,
            "similar_cases_text": similar_cases_text
        })

        return insight

    def get_case_count(self):
        """Get total number of cases diagnosed"""
        return len(self._load_cases())

    def display_statistics(self):
        """Display experience statistics"""
        cases = self._load_cases()

        if not cases:
            print("\n No cases in experience database yet.")
            return

        # Count diagnoses
        diagnoses = {}
        for case in cases:
            diagnosis = case['final_diagnosis'].split(":")[0].strip()
            diagnoses[diagnosis] = diagnoses.get(diagnosis, 0) + 1

        print("\n" + "=" * 60)
        print(" BAYMAX EXPERIENCE STATISTICS")
        print("=" * 60)
        print(f"Total cases diagnosed: {len(cases)}")
        print(f"\nMost common diagnoses:")

        for diagnosis, count in sorted(diagnoses.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  • {diagnosis}: {count} cases")

        print("=" * 60 + "\n")
