import os
import subprocess
import json
import re
from typing import TypedDict, Optional, List, Dict
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
load_dotenv()
from langsmith import utils
from langsmith import traceable
from typing import Annotated
import operator
utils.tracing_is_enabled()


model = ChatGroq(model=os.getenv("MODEL"), temperature=0, api_key=os.getenv("GROQ_API_KEY"))





class FileStructureState(TypedDict):
   
   """A TypedDict representing the state of the file structure generation
   process with attributes: srs_text (str), file_structure (Optional[List[str]]),
   file_descriptions (Optional[Dict[str, str]]), folder_path (str), error_log
   Optional[str]), retry_count (int), code_feedback (Optional[Dict[str, str]]),
   improvement_count (int)."""
   
   srs_text: Annotated[str, operator.add]
   
   file_structure: Optional[List[str]]
   file_descriptions: Optional[Dict[str, str]]
   folder_path: str
   error_log: Optional[str]
   retry_count: int
   code_feedback: Optional[Dict[str, str]]
   improvement_count: int

from json import JSONDecodeError


@traceable
def srs_to_file_structure(state: FileStructureState) -> FileStructureState:
 
    """Generates a file structure and descriptions from the SRS document."""
   
    prompt = f"""
    You are a software architect. Given the following SRS document:
    {state["srs_text"]}
    - Generate a structured JSON file tree.
    - Provide a detailed description of each file's purpose and what should be inside it and generate docker file as well and create readme files for every thing and requirements.txt.
    - do not generate tests
    -List all the required file and folder paths.
    -Follow best practices for modern FastAPI projects and better folder structure.
    - Return a JSON object with:
      - 'files': List of file paths.
      - 'descriptions': Dictionary mapping file paths to their descriptions. Each description should comprehensively outline the structure and purpose of the file, including: Classes:List all the classes that should be present in the file., Provide a detailed description of what each class should do.,Explain the role and functionality of each class within the context of the file.,Variables: List all the key variables that should be present in the file., Describe the purpose and usage of each variable., Include details on the scope and type of each variable., Methods: List all the methods that should be present in the file. Provide a detailed description of what each method should do., Explain the inputs, outputs, and side effects of each method.
    - Ensure the response is in valid JSON format without any additional text, markdown, or code blocks.
    """
   
    response = model.invoke(prompt)
    response_str = response.content
    with open("debug_response.json", "w") as debug_file:
        debug_file.write(response_str)
    # print("Raw Response:",response_str)
    
    # json_match = re.search(r"```json\s+(.*?)\s+```", response_str, re.DOTALL)
    try:
        json_match = re.search(r"```json\s+(.*?)\s+```", response_str, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            json_data = json.loads(json_str)
            try:
                json_data = json.loads(json_str)
            except JSONDecodeError as e:
                print(f"JSONDecodeError: {e}")
                print("Invalid JSON Response:", json_str)
                raise e
            state["file_structure"] = json_data["files"]
            state["file_descriptions"] = json_data["descriptions"]
        else:
            print("JSON data not found in the response.")
    except JSONDecodeError as e:
        print(f"JSONDecodeError: {e}")
        print("Response content:", response_str)
        raise e
 
    return state

@traceable
def create_files_tool(state: FileStructureState) -> FileStructureState:
 
    """Creates files and stores descriptions for the next step."""
   
    folder_path = state.get("folder_path", "generated_project")
    file_structure = state.get("file_structure", [])
    file_descriptions = state.get("file_descriptions", {})
 
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
 
    for file_path in file_structure:
        full_path = os.path.join(folder_path, file_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
 
        with open(full_path, "w") as f:
            f.write(f"# Description: {file_descriptions.get(file_path, 'No description available')}\n\n")
 
    return state

import os
import subprocess
import sys
from dotenv import load_dotenv
 
def write_code_to_files(state: dict) -> dict:
 
    """Writes code into the generated files based on descriptions and
    appends requirements to requirements.txt."""
 
    folder_path = state.get("folder_path", "generated_project")
    file_structure = state.get("file_structure", [])
    file_descriptions = state.get("file_descriptions", {})
 
    requirements = set()
 
    for file_path in file_structure:
        full_path = os.path.join(folder_path, file_path)
        description = file_descriptions.get(file_path, "")
        print(full_path, description)
 
        prompt = f"""
        You are a senior FastAPI developer. Generate a complete Python file based **only** on the following description:  
        {description}  
 
        ### **Constraints:**  
        - **Only generate code for this specific file:** {file_path}  
        - **Do not generate code for any other files.**  
        - **Strictly adhere to the extracted requirements from the description.**  
        - **Do not assume or add extra functionality beyond what is specified.**  
        - **Follow FastAPI best practices, keeping the code minimal yet correct.**  
        - **Use clear and concise variable and function names.**  
        - **Ensure modularity and error handling but avoid unnecessary abstractions.**  
        - **Include only relevant docstrings and comments.**  
        - **Do not generate unit tests unless explicitly requested.**  
        - **Do Not Give Anything Like pip installs and everything that is present should be python. Do not give notes as well.**
        """
 
        response = model.invoke(prompt)
        code = response.content.strip()
 
        code_lines = code.split('\n')
        filtered_code = "\n".join(line for line in code_lines if "```" not in line)
 
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w") as f:
            f.write(filtered_code)
 
        for line in code_lines:
            if "import " in line or "from " in line:
                parts = line.split()
                if parts[0] == "import":
                    requirement = parts[1].split('.')[0]
                elif parts[0] == "from":
                    requirement = parts[1].split('.')[0]
                if requirement not in file_structure and requirement != "main" and requirement not in file_descriptions:
                    requirements.add(requirement)
 
    requirements_path = os.path.join(folder_path, "requirements.txt")
    with open(requirements_path, "w") as req_file:
        for requirement in sorted(requirements):
            req_file.write(f"{requirement}\n")
 
    if os.path.exists(requirements_path):
        print("Installing dependencies...")
        try:
            result = subprocess.run([sys.executable, "-m", "pip", "install", "-r", requirements_path], check=True, capture_output=True, text=True)
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Failed to install dependencies: {e}")
 
    venv_path = os.path.join(folder_path, "venv")
    if not os.path.exists(venv_path):
        print("Creating virtual environment...")
        subprocess.check_call([sys.executable, "-m", "venv", venv_path])
 
    if os.name == "nt":
        activate_script = os.path.join(venv_path, "Scripts", "activate")
    else:
        activate_script = os.path.join(venv_path, "bin", "activate")
 
    env_path = os.path.join(folder_path, ".env")
    if not os.path.exists(env_path):
        print("Creating .env file...")
        with open(env_path, "w") as env_file:
            env_file.write("KEY=VALUE\n")  
           
    if os.path.exists(env_path):
        print("Loading environment variables...")
        load_dotenv(env_path)
 
    print("Environment setup complete.")
    return state
 



graph = StateGraph(FileStructureState)

workflow=graph.compile()

def read_extracted_text():
    with open("extracted_text.txt", "r") as f:
        return f.read()
srs_text_doc = read_extracted_text()

initial_state = {
    "srs_text": srs_text_doc,
    "file_structure": None,
    "file_descriptions": None,
    "folder_path": "generated_project",
    "error_log": None,
    "retry_count": 0,
    "code_feedback": None
}

workflow.invoke(initial_state)