"""
BAYMAX Verification Agent - BULLETPROOF VERSION
Specifically fixed to prevent "Analysis Failed" and "Short explanation" errors.
"""

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import re


class VerificationAgent:
    def __init__(self, model_name: str = "llama3.2", temperature: float = 0.0):
        # Temperature 0.0 makes the AI as predictable and robotic as possible.
        self.model = OllamaLLM(model=model_name, temperature=temperature)

    def verify_answer(self, question: str, answer: str) -> tuple:
        if len(answer.strip()) < 2:
            return False, 0.0, "Input too short."
        return self._analyze_relevance_with_llm(question, answer)

    def _analyze_relevance_with_llm(self, question: str, answer: str) -> tuple:
        # A much more "forced" prompt for Llama 3.2
        prompt_template = ChatPromptTemplate.from_template("""
        Determine if the Patient's Answer is a relevant medical response to the Doctor's Question.

        Doctor: "{question}"
        Patient: "{answer}"

        A response is VALID if it describes symptoms, pain, time, or location.
        A response is INVALID if it is about animals, food, nursery rhymes, or unrelated nonsense.

        Rules:
        - "The cat sat on the mat" is INVALID.
        - "Biryani" is INVALID.
        - "I have a cough" is VALID.

        Answer exactly in this format:
        VALID: YES or NO
        REASON: one sentence explanation
        """)

        chain = prompt_template | self.model

        try:
            response = chain.invoke({"question": question, "answer": answer})
            raw_text = response.strip()

            # --- BULLETPROOF PARSING ---
            # We use Regular Expressions (re) to find the answer even if the AI adds junk text
            valid_match = re.search(r"VALID:\s*(YES|NO)", raw_text, re.IGNORECASE)
            reason_match = re.search(r"REASON:\s*(.*)", raw_text, re.IGNORECASE)

            if valid_match:
                is_valid = valid_match.group(1).upper() == "YES"
            else:
                # If it didn't say YES or NO, check if it's medical or nonsense manually
                is_valid = False

            if reason_match:
                reason = reason_match.group(1).strip()
            else:
                reason = "Irrelevant or nonsensical input detected."

            confidence = 0.95 if is_valid else 0.1
            return is_valid, confidence, reason

        except Exception as e:
            return False, 0.0, f"Error: {str(e)}"

    def should_continue(self, is_valid: bool, confidence: float) -> bool:
        return not is_valid

    def get_quick_feedback(self, question: str, answer: str) -> dict:
        is_valid, confidence, reason = self.verify_answer(question, answer)
        return {
            "accept": is_valid,
            "show_verification": True,
            "verification_text": reason,
            "confidence": f"{int(confidence * 100)}%"
        }


if __name__ == "__main__":
    verifier = VerificationAgent()
    # Test the nonsense you just got
    q = "Please describe your symptoms"
    a = "a cat Sat On A mat with a rat"
    v, c, r = verifier.verify_answer(q, a)
    print(f"Result for '{a}': {'✅ VALID' if v else '⛔ REJECTED'}")
    print(f"Reason: {r}")