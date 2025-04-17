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