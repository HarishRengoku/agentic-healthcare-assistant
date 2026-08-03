"""
BAYMAX Medical Reference System - FIXED TRANSCRIPT VERSION
Only saves verified medical input to the final transcript.
"""

from diagnostic_agent import DiagnosticAgent
from experience_db import ExperienceDB
from voice_input_module import VoiceInputModule, InputModeSelector, VoiceLoggingSystem
from entity.verification_agent import VerificationAgent
import time
import subprocess


class BAYMAXDoctor:
    """BAYMAX with strict transcript filtering"""

    def __init__(self, enable_voice: bool = True):
        """Initialize BAYMAX"""
        self.agent = DiagnosticAgent()
        self.experience_db = ExperienceDB()
        self.verifier = VerificationAgent()

        self.enable_voice = enable_voice
        self.voice_module = None
        self.input_selector = None
        self.voice_logger = None

        if self.enable_voice:
            try:
                self.voice_module = VoiceInputModule(language="en-US")
                self.input_selector = InputModeSelector(self.voice_module)
                self.voice_logger = VoiceLoggingSystem()
                print("✓ Voice input enabled")
            except Exception as e:
                print(f"⚠️  Voice input disabled: {e}")
                self.enable_voice = False

    def speak_question(self, question: str):
        """Speak the question using PowerShell"""
        if self.enable_voice:
            try:
                text = question.replace('"', '\\"').replace('\n', ' ').strip()
                if not text:
                    return

                ps_cmd = (
                    'Add-Type -AssemblyName System.Speech; '
                    '$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; '
                    '$s.Rate = 0; '
                    '$s.Volume = 100; '
                    f'$s.Speak("{text}")'
                )

                subprocess.Popen(
                    ['powershell', '-NoProfile', '-Command', ps_cmd],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
                )
            except Exception as e:
                pass

    def get_voice_input_continuous(self, prompt: str, question: str = None) -> str:
        """
        Get voice input with continuous mic
        FIXED: Only commits 'Accepted' segments to the final history.
        """
        print("\n" + "─" * 70)
        print("🎤 VOICE INPUT MODE - Continuous Recording")
        print("─" * 70)
        print(f"\n{prompt}\n")
        print("📋 INSTRUCTIONS:")
        print("  • Type 'k' and press ENTER to OPEN microphone")
        print("  • Speak your answer (mic stays open)")
        print("  • Type 'k' and press ENTER again to CLOSE and submit")
        print("  • Type 'skip' to switch to text input")
        print("\n" + "─" * 70 + "\n")

        while True:
            user_input = input("Press 'k' to open mic, or type 'skip' to use text: ").strip().lower()

            if user_input == "skip":
                print("\n📝 TEXT INPUT MODE")
                print("─" * 70)
                text_response = input(f"{prompt}: ").strip()
                return text_response

            elif user_input == "k":
                break
            else:
                print("❌ Invalid input. Type 'k' to open mic or 'skip' to use text.\n")
                continue

        print("\n🎤 MICROPHONE OPEN")
        print("───────────────────────────────────────────────────────────────────")
        print("🔴 Recording... Speak your answer!")
        print("───────────────────────────────────────────────────────────────────")
        print("\n📝 Type 'k' and press ENTER when done speaking to close mic\n")

        # --- TRANSCRIPT FILTERING FIX ---
        valid_segments = []  # We store only verified text here
        reask_count = 0

        while True:
            try:
                print("🎤 Listening...\n")
                recognized_text = self.voice_module.listen(timeout=30, phrase_time_limit=30)

                if recognized_text:
                    print(f"\n✓ Recognized: {recognized_text}")

                    # Check validity using the Verification Agent
                    is_segment_valid = True
                    if question:
                        feedback = self.verifier.get_quick_feedback(question, recognized_text)

                        if feedback["show_verification"]:
                            print(f"✓ Verification: {feedback['verification_text']} ({feedback['confidence']})")

                        # If input is irrelevant
                        if not feedback["accept"]:
                            is_segment_valid = False
                            print(f"⚠️  Discarding irrelevant segment.")

                            # Allow one attempt to clarify if it was very short
                            if reask_count == 0 and len(recognized_text.strip()) < 25:
                                reask_count += 1
                                print(f"⚠️  Please try to be more specific.\n")
                                if "clarification" in feedback:
                                    print(f"Doctor: {feedback['clarification']}\n")
                                continue

                    # --- COMMIT LOGIC ---
                    if is_segment_valid:
                        valid_segments.append(recognized_text)
                        reask_count = 0  # Reset on successful valid input
                        print("✅ Added to medical context.")
                    else:
                        print("❌ Segment ignored (Nonsense/Food/Irrelevant).")

                    print("───────────────────────────────────────────────────────────────────")
                    print("Continue speaking or type 'k' to close mic and submit...\n")

                else:
                    print("\n❌ Could not understand audio. Continue speaking...\n")
                    continue

            except Exception as e:
                print(f"\n❌ Error: {e}\n")
                continue

            print("───────────────────────────────────────────────────────────────────")
            close_input = input(
                "Type 'k' to close mic and submit, or press ENTER to continue: ").strip().lower()

            if close_input == "k":
                if valid_segments:
                    final_text = " ".join(valid_segments)
                    print(f"\n✓ Microphone closed")
                    print(f"✓ Total verified transcript: {final_text}")
                    return final_text
                else:
                    print("\n❌ No valid medical speech was recorded yet. Keep talking...\n")
                    continue
            else:
                print("\n🎤 Listening again...\n")
                continue

    def print_header(self):
        """Display BAYMAX header"""
        total_cases = self.experience_db.get_case_count()
        experience_text = f"Experience: {total_cases} cases treated" if total_cases > 0 else "Experience: Learning mode"
        voice_status = "✓ Voice Input Enabled" if self.enable_voice else "⊘ Text Only Mode"

        header = f"""
╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║            🏥 BAYMAX MEDICAL REFERENCE SYSTEM 🏥             ║
║                                                                ║
║           "Hello, I'm BAYMAX, your medical assistant"         ║
║                                                                ║
║    {experience_text:<55}║
║                                                                ║
║    Knowledge Base:                                              ║
║    • 8 Medical Textbooks (Harrison's, Kumar & Clark's, etc)   ║
║    • Experience Database (Learning from past cases)            ║
║    • Online Search Tool (Real-time epidemiological data)      ║
║    • Smart Verification (Listens & validates answers)         ║
║    • {voice_status:<51}║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
"""
        print(header)

    def get_initial_symptoms(self) -> str:
        """Get initial symptoms from patient"""
        if self.enable_voice:
            self.input_selector.select_input_mode()
            print("\n" + "=" * 70)
            print("Let's begin. Please describe your symptoms.\n")
            print("=" * 70 + "\n")
            self.speak_question("Please describe what symptoms you're experiencing")

            if self.input_selector.input_mode == "voice":
                symptoms = self.get_voice_input_continuous(
                    "Please describe what symptoms you're experiencing",
                    question="What symptoms are you experiencing?"
                )
            else:
                symptoms = input("You: ").strip()
        else:
            print("\n" + "=" * 70 + "\nLet's begin. Please describe your symptoms.\n" + "=" * 70 + "\n")
            symptoms = input("You: ").strip()

        return symptoms

    def get_patient_response(self, doctor_question: str) -> str:
        """Get patient response"""
        self.speak_question(doctor_question)

        if self.enable_voice and self.input_selector.input_mode == "voice":
            response = self.get_voice_input_continuous(doctor_question, question=doctor_question)
            if self.voice_logger:
                self.voice_logger.log_voice_interaction(
                    audio_input=doctor_question,
                    recognized_text=response,
                    input_type="response"
                )
            return response
        else:
            print(f"\nDoctor (BAYMAX): {doctor_question}\n")
            response = input("You: ").strip()
            while not response:
                print("❌ Please provide an answer.")
                response = input("You: ").strip()
            return response

    def diagnose(self):
        """Run the diagnostic conversation"""
        self.print_header()
        initial_symptoms = self.get_initial_symptoms()

        if not initial_symptoms:
            print("Please describe your symptoms to continue.")
            return

        if self.enable_voice and self.voice_logger and self.input_selector.input_mode == "voice":
            self.voice_logger.log_voice_interaction("Initial symptom input", initial_symptoms, "symptom")

        print("\n" + "─" * 70)
        print("Doctor (BAYMAX): Where are you located?")
        print("─" * 70 + "\n")
        self.speak_question("Please tell me your location or region")

        if self.enable_voice and self.input_selector.input_mode == "voice":
            patient_region = self.get_voice_input_continuous("Please tell me your location", "Where are you located?")
        else:
            patient_region = input("You: ").strip()

        if patient_region:
            print(f"\n🌍 Analyzing epidemiology for {patient_region}...")
            time.sleep(1)

        similar_cases = self.agent.experience_db.find_similar_cases(initial_symptoms)
        if similar_cases:
            print("\n" + "💡" * 35 + "\n🧠 BAYMAX EXPERIENCE CHECK:")
            print(self.agent.experience_db.get_experience_insight(initial_symptoms, similar_cases))
            print("💡" * 35 + "\n")

        conversation_history = []
        print("\n" + "─" * 70 + "\nDoctor (BAYMAX): Interesting. Let me gather more information...\n" + "─" * 70)

        while True:
            next_question = self.agent.get_follow_up_question(initial_symptoms, conversation_history, patient_region)
            patient_answer = self.get_patient_response(next_question)

            if not patient_answer: continue
            if patient_answer.lower() in ["done", "exit", "q"]: break

            conversation_history.append((next_question, patient_answer))
            if not self.agent.should_continue_asking(initial_symptoms, conversation_history):
                print("Doctor (BAYMAX): I have enough information now.\n")
                break

        print("=" * 70 + "\n🔍 CLINICAL ANALYSIS & DIAGNOSIS\n" + "=" * 70)
        time.sleep(1)
        diagnosis = self.agent.narrow_down_diagnosis(initial_symptoms, conversation_history, patient_region)
        print(diagnosis)

        confidence = "High" if "High" in diagnosis else ("Medium" if "Medium" in diagnosis else "Low")
        self.agent.experience_db.store_case(initial_symptoms, conversation_history, diagnosis, confidence)

        if self.enable_voice:
            self.speak_question("I have completed my analysis. Please review the detailed diagnosis above.")
        print("\n" + "=" * 70 + "\n⚠️  DISCLAIMER: Educational reference only.\n" + "=" * 70)


def main():
    """Main entry point"""
    print("\n" + "=" * 70 + "\n🏥 BAYMAX MEDICAL REFERENCE SYSTEM - STARTUP\n" + "=" * 70)
    voice_enabled = input("Enable voice input? (yes/no) [default: yes]: ").strip().lower()
    enable_voice = voice_enabled != "no"

    baymax = BAYMAXDoctor(enable_voice=enable_voice)

    while True:
        try:
            baymax.diagnose()
            if input("\nSee another patient? (yes/no): ").strip().lower() != "yes":
                baymax.experience_db.display_statistics()
                print("\n👋 Thank you for using BAYMAX")
                break
        except KeyboardInterrupt:
            print("\n👋 Thank you for using BAYMAX")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            time.sleep(2)

if __name__ == "__main__":
    main()