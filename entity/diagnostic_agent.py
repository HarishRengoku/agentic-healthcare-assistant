"""
BAYMAX Diagnostic Agent (UPGRADED WITH ONLINE SEARCH)
Now searches the web for epidemiological data!
"""

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from retrieval_tool import MedicalRetriever
from experience_db import ExperienceDB
from online_search_tool import OnlineSearchTool
# import json


class DiagnosticAgent:
    """learns from experience AND searches online"""

    def __init__(self):
        self.retriever = MedicalRetriever(db_location="./medical_langchain_db", k=10)
        self.model = OllamaLLM(model="llama3.2", temperature=0.4)
        self.experience_db = ExperienceDB()
        self.search_tool = OnlineSearchTool()  # NEW: Online search
        self.conversation_history = []
        self.patient_region = None  # Track patient region

    def get_follow_up_question(self, symptoms, previous_answers, patient_region=None):
        """Generate next question, considering past experience and online data"""

        self.patient_region = patient_region

        conversation_so_far = ""
        for i, (q, a) in enumerate(previous_answers, 1):
            conversation_so_far += f"Q{i}: {q}\nA{i}: {a}\n\n"

        # Check if we've seen similar cases before
        similar_cases = self.experience_db.find_similar_cases(symptoms)
        experience_context = ""

        if similar_cases:
            experience_context = f"\n\nRELEVANT PAST CASES: You have {len(similar_cases)} similar case(s) in your experience."
            experience_context += "\n".join([
                f"- Case #{case['case_id']}: Diagnosed as {case['final_diagnosis'].split(':')[0].strip()}"
                for case in similar_cases
            ])

        # NEW: Get online epidemiological context if region provided
        online_context = ""
        if patient_region:
            online_context = f"\n\nREGIONAL CONTEXT: {patient_region}"
            online_context += "\n(Searching for endemic diseases and common pathogens in this region...)"

        prompt_template = ChatPromptTemplate.from_template("""
You are a skilled clinical doctor. A patient has described their initial symptoms and you've been gathering information.

INITIAL SYMPTOMS: {symptoms}

CONVERSATION SO FAR:
{conversation}
{experience_context}
{online_context}

YOUR TASK: Based on what you know so far, ask the NEXT most important diagnostic question to narrow down the diagnosis.

If you haven't already, consider asking about:
- Geographic location/travel history
- Recent exposures unique to their region
- Endemic diseases in their area

Be SPECIFIC and CLINICAL. Ask about:
- Timing/onset
- Associated symptoms
- Recent exposures
- Medical history
- Red flag symptoms

Ask ONLY ONE focused question. Make it sound natural, like a real doctor.
Format: Just the question, nothing else.
""")

        chain = prompt_template | self.model
        next_question = chain.invoke({
            "symptoms": symptoms,
            "conversation": conversation_so_far if conversation_so_far else "No previous questions yet.",
            "experience_context": experience_context,
            "online_context": online_context
        })

        return next_question.strip()

    def narrow_down_diagnosis(self, symptoms, conversation_history, patient_region=None):
        """Generate diagnosis with textbook + experience + online epidemiology"""

        # Build conversation string
        full_conversation = f"Initial Symptoms: {symptoms}\n\n"
        for q, a in conversation_history:
            full_conversation += f"Doctor: {q}\nPatient: {a}\n\n"

        # Get relevant books
        query = symptoms + " " + " ".join([a for _, a in conversation_history])
        docs = self.retriever.retrieve(query)
        knowledge = self.retriever.format_knowledge(docs)

        # Get similar past cases
        similar_cases = self.experience_db.find_similar_cases(symptoms, top_k=3)
        past_experience = ""

        if similar_cases:
            past_experience = "\n\nPAST SIMILAR CASES FROM EXPERIENCE:"
            for case in similar_cases:
                past_experience += f"\n- Case #{case['case_id']}: {case['initial_symptoms']} → {case['final_diagnosis']} ({case['confidence']})"

        # NEW: Get online epidemiological data
        online_epidemiology = ""
        if patient_region:
            online_epidemiology = f"\n\nONLINE EPIDEMIOLOGICAL DATA FOR {patient_region}:"
            try:
                # Search for relevant pathogens and diseases
                regional_pathogens = self.search_tool.search_regional_pathogens(patient_region)
                online_epidemiology += "\n" + regional_pathogens
            except:
                online_epidemiology += "\n(Online search unavailable)"

        prompt_template = ChatPromptTemplate.from_template("""
You are a master diagnostician with growing experience and access to real-time epidemiological data.
Based on this patient conversation, textbook knowledge, past cases, and regional epidemiology,
determine the MOST LIKELY diagnosis.

PATIENT CONVERSATION:
{conversation}

TEXTBOOK KNOWLEDGE:
{knowledge}
{past_experience}
{online_epidemiology}

TASK: Based on clinical reasoning from ALL sources (textbooks, experience, epidemiology):

1. **MOST LIKELY DIAGNOSIS**: [diagnosis]
2. **CONFIDENCE**: [High/Medium/Low]
3. **KEY SUPPORTING FEATURES**: [specific features that point to this]
4. **REGIONAL CONSIDERATIONS**: [how regional epidemiology affects diagnosis]
5. **WHAT RULES IT OUT**: [what would make this diagnosis unlikely]
6. **NEXT DIAGNOSTIC TESTS**: [specific tests]
7. **IMMEDIATE MANAGEMENT**: [what to do now]
8.  **MEDICINES TO BE PRESCRIBED**: [what antibiotics, medications to take, what other items that need to prescribed to the patient to consume such as supplements, electrolytes, pain killers or changes in diet that need to bbe made etc etc]
9. **RED FLAGS**: [when to call emergency]

Consider regional prevalence when ranking differential diagnosis.
Be specific, evidence-based, and cite textbooks and epidemiological data where relevant.
""")

        chain = prompt_template | self.model
        diagnosis = chain.invoke({
            "conversation": full_conversation,
            "knowledge": knowledge,
            "past_experience": past_experience,
            "online_epidemiology": online_epidemiology
        })

        return diagnosis

    def should_continue_asking(self, symptoms, conversation_history):
        """Decide if we have enough info"""

        if len(conversation_history) >= 5:
            return False

        if len(conversation_history) >= 2:
            prompt_template = ChatPromptTemplate.from_template("""
Based on this conversation, do we have ENOUGH information to make a diagnosis?

SYMPTOMS: {symptoms}
QUESTIONS ASKED: {num_questions}
CONVERSATION:
{conversation}

Answer with ONLY "YES" or "NO".
""")

            conversation_str = "\n".join([f"Q: {q}\nA: {a}" for q, a in conversation_history])

            chain = prompt_template | self.model
            response = chain.invoke({
                "symptoms": symptoms,
                "num_questions": len(conversation_history),
                "conversation": conversation_str
            })

            return "NO" in response.upper()

        return True
