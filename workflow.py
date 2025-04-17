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