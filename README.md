# BAYMAX – Agentic Healthcare Assistant

BAYMAX is a Python CLI-based medical reference assistant developed as a final-year academic project. It runs through the PyCharm console and uses locally hosted AI models, medical textbook retrieval, regional health information, and interactive follow-up questions to produce structured diagnostic guidance.

> **Disclaimer:** BAYMAX is an educational prototype and is **not** a medical device. Its output must not replace professional medical advice, diagnosis, prescriptions, or emergency care.

---

## Features

- Interactive symptom consultation through the Python console
- Text and optional microphone input
- Locally hosted Ollama language models
- Medical textbook retrieval using LangChain and ChromaDB
- AI-generated follow-up questions
- Response verification
- Regional epidemiological web search
- Previous-case memory
- Structured diagnostic reference output
- Local text-to-speech support

---

## Technology

- Python
- Ollama
- Llama 3.2
- `mxbai-embed-large`
- LangChain
- ChromaDB
- SpeechRecognition
- PyAudio
- DuckDuckGo Search

---

## Project Structure

```text
agentic-healthcare-assistant/
├── entity/
│   ├── baymax.py
│   ├── diagnostic_agent.py
│   ├── verification_agent.py
│   ├── retrieval_tool.py
│   ├── experience_db.py
│   ├── online_search_tool.py
│   ├── database.py
│   ├── train_baymax.py
│   ├── test_agent.py
│   ├── quick_eval_diagnostic.py
│   ├── evaluate_chatbot_full.py
│   └── evaluate_transformers_metrics.py
├── medical_langchain_db/
├── baymax_experiences/
│   └── cases.json
├── voice_input_module.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Prerequisites

Install the following software before running BAYMAX:

- Python 3.13
- PyCharm
- Git and Git LFS
- Ollama

---

## Installation

Clone the repository:

```powershell
git lfs install
git clone https://github.com/HarishRengoku/agentic-healthcare-assistant.git
cd agentic-healthcare-assistant
```

Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install the required Python packages:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Download the required Ollama models:

```powershell
ollama pull llama3.2
ollama pull mxbai-embed-large
```

Ensure the Ollama service is running before starting BAYMAX.

---

## Running in PyCharm

1. Open the cloned repository as a PyCharm project.
2. Select the project's Python interpreter from `.venv`.
3. Create a new Python Run Configuration.
4. Set the script path to:

   ```text
   entity/baymax.py
   ```

5. Set the working directory to the project root.
6. Enable **Add content roots to PYTHONPATH**.
7. Run the configuration.
8. Choose whether to enable voice input and follow the prompts in PyCharm's Run console.

---

## Runtime Data

Completed consultations are stored locally in:

```text
baymax_experiences/cases.json
```

Do **not** commit real patient information. Use only synthetic or properly anonymized demonstration data.

The medical textbook vector database is stored in:

```text
medical_langchain_db/
```

Because it is large, it is tracked using **Git LFS**.

---

## Evaluation Scripts

The `entity` directory also contains scripts for testing and evaluating the assistant:

- `quick_eval_diagnostic.py`
- `evaluate_chatbot_full.py`
- `evaluate_transformers_metrics.py`
- `test_agent.py`
- `train_baymax.py`

These scripts are intended for development and evaluation and are **not** required for normal consultations.

---

## Limitations

- AI-generated medical information may be incomplete or inaccurate.
- Web search requires an active internet connection.
- Voice recognition quality depends on the microphone and surrounding environment.
- Ollama models require sufficient local storage and system memory.
- BAYMAX is intended solely for local academic demonstration and research purposes.

---

## Responsible Use

Do **not** use BAYMAX for emergencies, autonomous diagnosis, or unsupervised treatment decisions.

Always consult a qualified healthcare professional for medical concerns, and contact your local emergency services immediately in urgent or life-threatening situations.
