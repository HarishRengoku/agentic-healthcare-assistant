# Voice Input Module for BAYMAX
# Implements Speech-to-Text for patient symptom input

import speech_recognition as sr
import pyttsx3
from typing import Optional, Tuple
import os
import json
from datetime import datetime


class VoiceInputModule:
    """Handle voice input and text-to-speech output for BAYMAX"""

    def __init__(self, language: str = "en-US"):
        """
        Initialize voice input/output module
        
        Args:
            language: Language code for speech recognition (e.g., 'en-US', 'en-GB', 'es-ES')
        """
        # Initialize speech recognizer
        self.recognizer = sr.Recognizer()
        
        # Initialize text-to-speech engine
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)  # Speech rate (words per minute)
        self.tts_engine.setProperty('volume', 0.9)  # Volume level (0-1)
        
        # Set language
        self.language = language
        
        # Microphone setup
        self.microphone = sr.Microphone()
        
        # Adjust recognizer for ambient noise (improves accuracy)
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
        
        # Voice preferences
        self.voices = self.tts_engine.getProperty('voices')
        if len(self.voices) > 0:
            # Use default voice (typically first one)
            self.tts_engine.setProperty('voice', self.voices[0].id)

    def speak(self, text: str, wait_for_completion: bool = True):
        """
        Convert text to speech
        
        Args:
            text: Text to speak
            wait_for_completion: If True, wait for speech to finish
        """
        try:
            self.tts_engine.say(text)
            if wait_for_completion:
                self.tts_engine.runAndWait()
        except Exception as e:
            print(f"❌ Text-to-speech error: {e}")

    def listen(self, timeout: int = 10, phrase_time_limit: int = 30) -> Optional[str]:
        """
        Listen to microphone and convert speech to text
        
        Args:
            timeout: Timeout in seconds to wait for speech
            phrase_time_limit: Maximum duration of phrase in seconds
            
        Returns:
            Recognized text or None if unsuccessful
        """
        try:
            with self.microphone as source:
                print("\n🎤 Listening... (speak now)")
                
                # Listen with timeout
                audio = self.recognizer.listen(
                    source, 
                    timeout=timeout,
                    phrase_time_limit=phrase_time_limit
                )
            
            print("🔄 Processing audio...")
            
            # Try Google Speech Recognition (free, good accuracy)
            try:
                text = self.recognizer.recognize_google(audio, language=self.language)
                print(f"✓ Recognized: {text}")
                return text
                
            except sr.UnknownValueError:
                print("❌ Could not understand audio. Please speak clearly and try again.")
                return None
            except sr.RequestError as e:
                print(f"❌ Speech recognition service error: {e}")
                print("   (Make sure you have internet connection)")
                return None
                
        except sr.WaitTimeoutError:
            print("❌ No speech detected. Please try again.")
            return None
        except Exception as e:
            print(f"❌ Microphone error: {e}")
            return None

    def get_voice_input(self, prompt: str = "", max_retries: int = 3) -> str:
        """
        Get voice input from user with retry logic
        
        Args:
            prompt: Text to speak before listening
            max_retries: Maximum number of retry attempts
            
        Returns:
            Recognized text from user
        """
        if prompt:
            self.speak(prompt)
        
        for attempt in range(max_retries):
            recognized_text = self.listen()
            
            if recognized_text:
                return recognized_text
            
            if attempt < max_retries - 1:
                retry_msg = f"Let's try again (attempt {attempt + 2}/{max_retries})"
                self.speak(retry_msg)
        
        print(f"\n❌ Could not get voice input after {max_retries} attempts.")
        print("Please enter text input instead:")
        return input("You: ").strip()


class InputModeSelector:
    """Allow user to choose between voice and text input"""

    def __init__(self, voice_module: VoiceInputModule):
        """Initialize input selector with voice module"""
        self.voice_module = voice_module
        self.input_mode = "text"  # Default mode

    def select_input_mode(self) -> str:
        """
        Let user choose input mode (voice or text)
        
        Returns:
            Selected mode: 'voice' or 'text'
        """
        print("\n" + "=" * 70)
        print("INPUT MODE SELECTION")
        print("=" * 70)
        print("1. 🎤 Voice Input (Speak your symptoms)")
        print("2. ⌨️  Text Input (Type your symptoms)")
        print("=" * 70 + "\n")
        
        while True:
            choice = input("Select mode (1 or 2): ").strip()
            
            if choice == "1":
                self.input_mode = "voice"
                print("\n✓ Voice mode selected. Make sure your microphone is working.\n")
                self.voice_module.speak("Voice mode activated. Please speak clearly.")
                return "voice"
            elif choice == "2":
                self.input_mode = "text"
                print("\n✓ Text mode selected.\n")
                return "text"
            else:
                print("❌ Invalid choice. Please enter 1 or 2.")

    def get_initial_symptoms(self, prompt: str = "Describe your symptoms") -> str:
        """
        Get initial symptoms in selected mode
        
        Args:
            prompt: Prompt text to show/speak
            
        Returns:
            User's symptom description
        """
        if self.input_mode == "voice":
            return self._get_voice_symptoms(prompt)
        else:
            return self._get_text_symptoms(prompt)

    def _get_voice_symptoms(self, prompt: str) -> str:
        """Get symptoms via voice"""
        print(f"\n{prompt}")
        return self.voice_module.get_voice_input(
            prompt=f"Please describe your symptoms. {prompt}",
            max_retries=2
        )

    def _get_text_symptoms(self, prompt: str) -> str:
        """Get symptoms via text"""
        print(f"\n{prompt}")
        return input("You: ").strip()

    def get_follow_up_response(self, doctor_question: str) -> str:
        """
        Get response to a follow-up question
        
        Args:
            doctor_question: The doctor's question
            
        Returns:
            Patient's answer
        """
        if self.input_mode == "voice":
            print(f"\nDoctor (BAYMAX): {doctor_question}\n")
            response = self.voice_module.get_voice_input(
                prompt=doctor_question,
                max_retries=2
            )
            # Confirm what was heard
            print(f"\n[You said: {response}]")
            return response
        else:
            print(f"\nDoctor (BAYMAX): {doctor_question}\n")
            return input("You: ").strip()


class VoiceLoggingSystem:
    """Log voice interactions for quality improvement and debugging"""

    def __init__(self, log_file: str = "voice_interactions.json"):
        """Initialize voice logging system"""
        self.log_file = log_file
        self.interactions = self._load_interactions()

    def _load_interactions(self) -> list:
        """Load existing interactions from log"""
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r') as f:
                    return json.load(f)
            except:
                return []
        return []

    def log_voice_interaction(self, 
                            audio_input: str,
                            recognized_text: str,
                            confidence: float = None,
                            input_type: str = "symptom"):
        """
        Log a voice interaction
        
        Args:
            audio_input: Description of audio input
            recognized_text: Text recognized from speech
            confidence: Confidence score (0-1)
            input_type: Type of input (symptom, response, etc)
        """
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "audio_input": audio_input,
            "recognized_text": recognized_text,
            "confidence": confidence,
            "input_type": input_type
        }
        
        self.interactions.append(interaction)
        self._save_interactions()

    def _save_interactions(self):
        """Save interactions to log file"""
        try:
            with open(self.log_file, 'w') as f:
                json.dump(self.interactions, f, indent=2)
        except Exception as e:
            print(f"⚠️  Could not save voice log: {e}")

    def get_statistics(self) -> dict:
        """Get voice interaction statistics"""
        total = len(self.interactions)
        return {
            "total_interactions": total,
            "symptom_inputs": sum(1 for i in self.interactions if i["input_type"] == "symptom"),
            "response_inputs": sum(1 for i in self.interactions if i["input_type"] == "response")
        }


# ==================== INSTALLATION GUIDE ====================
# 
# Required packages:
# pip install SpeechRecognition
# pip install pyttsx3
# pip install pyaudio  # For microphone support
#
# Windows users: You may need to install pyaudio via:
# pip install pipwin
# pipwin install pyaudio
#
# macOS users: Install via Homebrew:
# brew install portaudio
# pip install pyaudio
#
# Linux users:
# sudo apt-get install portaudio19-dev
# pip install pyaudio
#
# ============================================================

if __name__ == "__main__":
    # Test the voice module
    print("\n" + "=" * 70)
    print("🎤 BAYMAX VOICE INPUT MODULE TEST")
    print("=" * 70 + "\n")
    
    try:
        # Initialize voice module
        voice = VoiceInputModule(language="en-US")
        voice.speak("Welcome to BAYMAX Voice Input Test. Speak after the beep.")
        
        # Get voice input
        result = voice.listen(timeout=5)
        
        if result:
            print(f"\n✓ Input received: {result}")
        else:
            print("\n❌ No input received")
            
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        print("\nMake sure to install required packages:")
        print("pip install SpeechRecognition pyttsx3 pyaudio")
