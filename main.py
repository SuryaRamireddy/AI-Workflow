from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from docx import Document
import io
import shutil
from pathlib import Path
 
app = FastAPI()
 
 
 
@app.get("/")
async def Working():
 
    """Uses This Api To Check If The Apis Work"""
 
    return "I Am Working Great"
 
 
 
@app.post("/upload/")
async def upload_docx(file: UploadFile = File(...)):
 
    """
    Takes a .docx file as input and returns the text from the document.
    Returns:
        str: The text extracted from the .docx document.
    """
 
    if file.content_type != "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return JSONResponse(content={"error": "Invalid file type"}, status_code=400)
    contents = await file.read()
    document = Document(io.BytesIO(contents))
    doc_text = []
    for paragraph in document.paragraphs:
        doc_text.append(paragraph.text)
    text_content = "\n".join(doc_text)
    text = {"content": "\n".join(doc_text)}
    with open("extracted_text.txt", "w") as f:
        f.write(text_content)
    return text
 
