\# BAYMAX – Agentic Healthcare Assistant



BAYMAX is a Python CLI-based medical reference assistant developed as a final-year academic project. It runs through the PyCharm console and uses locally hosted AI models, medical textbook retrieval, regional health information and interactive follow-up questions to produce structured diagnostic guidance.



> \*\*Disclaimer:\*\* BAYMAX is an educational prototype, not a medical device. Its output must not replace professional medical advice, diagnosis, prescriptions or emergency care.



\## Features



\- Interactive symptom consultation through the Python console

\- Text and optional microphone input

\- Locally hosted Ollama language models

\- Medical textbook retrieval using LangChain and ChromaDB

\- Follow-up question generation

\- Answer verification

\- Regional epidemiological web search

\- Previous-case memory

\- Structured diagnostic reference output

\- Local text-to-speech support



\## Technology



\- Python

\- Ollama

\- Llama 3.2

\- `mxbai-embed-large`

\- LangChain

\- ChromaDB

\- SpeechRecognition

\- PyAudio

\- DuckDuckGo Search



\## Project structure



```text

agentic-healthcare-assistant/

├── entity/

│   ├── baymax.py

│   ├── diagnostic\_agent.py

│   ├── verification\_agent.py

│   ├── retrieval\_tool.py

│   ├── experience\_db.py

│   └── online\_search\_tool.py

├── medical\_langchain\_db/

├── baymax\_experiences/

│   └── cases.json

├── voice\_input\_module.py

├── requirements.txt

└── README.md

```



\## Prerequisites



Install:



\- Python 3.13

\- PyCharm

\- Git and Git LFS

\- Ollama



\## Installation



Clone the repository using Git LFS:



```powershell

git lfs install

git clone https://github.com/HarishRengoku/agentic-healthcare-assistant.git

cd agentic-healthcare-assistant

```



Create a virtual environment:



```powershell

python -m venv .venv

.venv\\Scripts\\Activate.ps1

```



Install the Python dependencies:



```powershell

python -m pip install --upgrade pip

python -m pip install -r requirements.txt

```



Install the required Ollama models:



```powershell

ollama pull llama3.2

ollama pull mxbai-embed-large

```



Ensure Ollama is running before starting BAYMAX.



\## Running in PyCharm



1\. Open the cloned repository as a PyCharm project.

2\. Select the Python interpreter from `.venv`.

3\. Create a Python run configuration.

4\. Set the script path to `entity\\baymax.py`.

5\. Set the working directory to the repository root.

6\. Enable \*\*Add content roots to PYTHONPATH\*\*.

7\. Run the configuration.

8\. Choose whether to enable voice input and follow the prompts in PyCharm’s Run console.



\## Runtime data



Completed consultations are stored locally in:



```text

baymax\_experiences/cases.json

```



Do not commit real patient information. Only synthetic or properly anonymized demonstration data should be used.



The medical textbook vector database is stored with Git LFS under:



```text

medical\_langchain\_db/

```



\## Evaluation scripts



The `entity` directory also contains scripts for evaluating and testing the assistant:



\- `quick\_eval\_diagnostic.py`

\- `evaluate\_chatbot\_full.py`

\- `evaluate\_transformers\_metrics.py`

\- `test\_agent.py`

\- `train\_baymax.py`



These scripts are not required for a normal consultation.



\## Limitations



\- AI-generated medical information can be incomplete or incorrect.

\- Web-search functionality requires an internet connection.

\- Voice recognition quality depends on the microphone and environment.

\- Ollama models require sufficient local storage and memory.

\- The application is intended for local academic demonstration only.



\## Responsible use



Do not use BAYMAX for emergencies, autonomous diagnosis or unsupervised treatment decisions. Consult a qualified healthcare professional for medical concerns and contact local emergency services when urgent care is required.

