#!/usr/bin/env python

"""
Usage:
  rqa.py import [--api_key=<api_key>] [-b] <input_pdf> <vector_db>
  rqa.py search [-n <top_k>] <vector_db> <question>
  rqa.py ask [--api_key=<api_key>] <vector_db> <question>
  rqa.py quiz [--api_key=<api_key>] <vector_db> <questions_json> <output_json>

Commands:
  import      Extract text from a PDF, chunk it, and store it in the vector database.
  search      Retrieve relevant chunks from the vector database based on a question.
  ask         Query OpenAI with retrieved chunks to generate an answer.
  quiz        Generate responses for a set of questions and save them to a JSON file.


Options:
  -b                    Do not use semantic chunking on import [Default: False]
  --api_key=<api_key>   OpenAI API key [default: os.getenv("OPENAI_API_KEY")]
  -n <top_k>            Number of top chunks to retrieve [default: 10]

Arguments:
  <input_pdf>       Path to the input PDF document or a path to a folder of PDF documents.
  <question>        The question to ask.
  <vector_db>       Path to the ChromaDB vector database folder. Created with `import`.
  <questions_json>  JSON file containing questions for the quiz.
  <output_json>     Path to the output JSON file for quiz results.

"""

import os
import tiktoken
import chromadb
import openparse
import json

from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from docopt import docopt
from openai import OpenAI

from googleapiclient.discovery import build
from google.oauth2 import service_account


LOCAL_EMBEDDING_MODEL = "all-mpnet-base-v2"  # Local Embedding Model (~ 400 MB)
OPENAI_MODEL = "gpt-4o-mini"                 # OpenAI Model
MAX_TOKENS = 1024                            # Maximum tokens allowed for embeddings


##
## Need to process tables and images
## Need to process sequences
##
## table_args={
#                 "parsing_algorithm": "table-transformers",
#                 "min_table_confidence": 0.8,
#             }            



def extract_pdf_text(pdf_path, api_key=None):
    """Extract text from a PDF while preserving document structure."""
    if api_key:
        pipeline = openparse.processing.SemanticIngestionPipeline(
            openai_api_key=api_key,
            model="text-embedding-3-large",
            min_tokens=64,
            max_tokens=1024
        )
    else:
        pipeline = openparse.processing.BasicIngestionPipeline(
        )
    parser = openparse.DocumentParser(processing_pipeline=pipeline)
    return parser.parse(pdf_path)

def create_chunks(parsed_doc):
    """
    Create text chunks from the parsed document
    """
    chunks = []
    for i, node in enumerate(parsed_doc.nodes):
        chunks.append(f"<<FILENAME: {parsed_doc.filename}, PAGE: {node.start_page}, CHUNK: {i}>>\n{node.text}")
    return chunks

def initialize_chroma(vector_db_path):
    """Initialize the ChromaDB collection for storing embeddings."""
    client = chromadb.PersistentClient(path=vector_db_path)
    embedding_function = SentenceTransformerEmbeddingFunction(model_name=LOCAL_EMBEDDING_MODEL)
    collection = client.get_or_create_collection(name="document_chunks", embedding_function=embedding_function)
    return collection

def store_chunks_in_chroma(chunks, vector_db_path):
    """Stores adaptive, semantic-chunked document text into ChromaDB using local embeddings."""
    collection = initialize_chroma(vector_db_path)

    # Load the sentence-transformers model for local embeddings
    model = SentenceTransformer(LOCAL_EMBEDDING_MODEL)

    # Compute embeddings locally
    embeddings = model.encode(chunks, convert_to_numpy=True)

    # Insert chunks into ChromaDB with metadata
    for i, chunk in enumerate(chunks):
        collection.add(
            ids=[str(i)],
            documents=[chunk],
            embeddings=[embeddings[i].tolist()],  # Convert numpy to list
            metadatas=[{"chunk_id": i}]
        )
    
def find_relevant_chunks(question, vector_db_path, top_k=20):
    """Finds the most relevant chunks for a given question using ChromaDB."""
    collection = initialize_chroma(vector_db_path)

    # Load sentence-transformers model
    model = SentenceTransformer(LOCAL_EMBEDDING_MODEL)
    question_embedding = model.encode(question).tolist()

    # Query ChromaDB for the most relevant chunks
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k
    )
    return results["documents"][0] if "documents" in results else []

def make_prompt(question, relevant_chunks):
    """Creates a prompt for OpenAI based on the question and relevant document chunks."""
    return f"""

        You are an AI assistant answering a question based on a set of
        documents.  The relevant document sections are provided below
        where each document section is preceeded by `<<FILENAME: F,
        PAGE: X, CHUNK: C>>` where F is the filename X is the page
        number and C is the chunk ID for the subsequent text.
        
        Provide your answer in the following format:
            
        Answer: <answer>

        Evidence: <evidence>


        Where: 

            <answer> - should be a concise answer to the question
                       based only on the document.
    

            <evidence> - should be a list of relevant document text,
                        quoted verbatim, along with its FILENAME, PAGE
                        and CHUNK of origin supporting the answer.

                        
        **Relevant Document Sections:**

        {"\n\n".join(relevant_chunks)}

        **Question:** 

        {question}
        """

def query_openai(api_key, question, relevant_chunks):
    """Queries OpenAI with the most relevant document chunks."""
    client = OpenAI()
    prompt = make_prompt(question, relevant_chunks)
    completion = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=1024
    )
    answer = completion.choices[0].message.content
    return (prompt, answer)

def write_list_of_dicts_to_google_sheet(data_list, sheet_name, credentials_path="credentials.json"):
    """
    Writes a list of dictionaries to a new Google Sheet, resizes columns, and enables word wrap.

    :param data_list: List of dictionaries where keys are column headers and values are row values.
    :param sheet_name: Name of the Google Sheet to be created.
    :param credentials_path: Path to the Google service account credentials JSON file.
    :return: URL of the created Google Sheet.
    """
    if not data_list:
        raise ValueError("Data list cannot be empty")

    # Authenticate with Google Sheets API
    creds = service_account.Credentials.from_service_account_file(
        credentials_path, scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    )
    sheets_service = build("sheets", "v4", credentials=creds)
    drive_service = build("drive", "v3", credentials=creds)

    # Create a new spreadsheet
    spreadsheet = sheets_service.spreadsheets().create(
        body={"properties": {"title": sheet_name}},
        fields="spreadsheetId"
    ).execute()
    sheet_id = spreadsheet["spreadsheetId"]

    # Extract headers and rows
    headers = list(data_list[0].keys())
    values = [headers] + [[entry[key] for key in headers] for entry in data_list]

    # Write data to the sheet
    sheets_service.spreadsheets().values().update(
        spreadsheetId=sheet_id,
        range="A1",
        valueInputOption="RAW",
        body={"values": values}
    ).execute()

    # Resize columns and enable word wrap
    requests = [
        {
            "autoResizeDimensions": {
                "dimensions": {
                    "sheetId": 0,
                    "dimension": "COLUMNS",
                    "startIndex": 0,
                    "endIndex": len(headers),
                }
            }
        },
        {
            "repeatCell": {
                "range": {
                    "sheetId": 0,
                    "startRowIndex": 0,
                    "startColumnIndex": 0,
                    "endColumnIndex": len(headers),
                },
                "cell": {
                    "userEnteredFormat": {
                        "wrapStrategy": "WRAP",
                        "horizontalAlignment": "LEFT",
                        "verticalAlignment": "TOP",
                    }
                },
                "fields": "userEnteredFormat(horizontalAlignment,verticalAlignment,wrapStrategy)"
            }
        }
    ]

    sheets_service.spreadsheets().batchUpdate(
        spreadsheetId=sheet_id,
        body={"requests": requests}
    ).execute()

    drive_service.permissions().create(
        fileId=sheet_id,
        body={
            "type": "user",
            "role": "writer",
            "emailAddress": "jameshudsonbullard@gmail.com"
        }).execute()
    
    sheet_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}"

    return sheet_url

def main():
    args = docopt(__doc__)
    
    vector_db_path = args["<vector_db>"]
    # Disable tokenizers parallelism to avoid warnings.
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    if args["search"]:
        question = args["<question>"]
        top_k = int(args["-n"])
        relevant_chunks = find_relevant_chunks(question, vector_db_path, top_k)
        for chunk in relevant_chunks:
            print(f"chunk: {chunk}")
    else:
        api_key = args["--api_key"] if args["--api_key"] else os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Error: --api_key is required for 'import, ask and quiz' commands.")
            return
    
        if args["import"]:
            input_pdf = args["<input_pdf>"]
            basic_chunking = args["-b"]
            if basic_chunking:
                api_key = None
            
            if os.path.isdir(input_pdf):
                pdfs = set(os.path.join(input_pdf, f) for f in os.listdir(input_pdf) if f.endswith(".pdf"))
            else:
                pdfs = [input_pdf]
            
            print(f"Processing {len(pdfs)} pdf files.")
            
            for pdf in pdfs:
                print(f"Processing {pdf} file.")
                parsed_doc = extract_pdf_text(pdf, api_key)
                chunks = create_chunks(parsed_doc)
                store_chunks_in_chroma(chunks, vector_db_path)
                print(f"Stored {len(chunks)} chunks in ChromaDB at {vector_db_path}")
                

        elif args["ask"]:
            question = args["<question>"]
            api_key = args["--api_key"] if args["--api_key"] else os.environ["OPENAI_API_KEY"]
            if not api_key:
                print("Error: --api_key is required for 'ask' command.")
                return
            relevant_chunks = find_relevant_chunks(question, vector_db_path, top_k=4)
            prompt, answer = query_openai(api_key, question, relevant_chunks)
            print(f"Answer: {answer}\n\n{"-"*50}\n\nPrompt: {prompt}")

        elif args["quiz"]:
            questions_json = args["<questions_json>"]
            output_json = args["<output_json>"]
            vector_db_path = args["<vector_db>"]
            api_key = args["--api_key"] if args["--api_key"] else os.environ["OPENAI_API_KEY"]
            if not api_key:
                print("Error: --api_key is required for 'ask' command.")
                return
            
            qanda = []
            with open(questions_json, "r") as f:
                questions = json.load(f)
                for i, question in enumerate(questions):
                    print(f"Processing question: {i}")
                    qtxt = question["Question"]
                    try:
                        relevant_chunks = find_relevant_chunks(qtxt, vector_db_path, top_k=4)
                        prompt, answer = query_openai(api_key, qtxt, relevant_chunks)
                    except Exception as e:
                        print(f"Error processing question: {i}, due to exception: {e}")
                        answer = f"Exception: {e}"

                    # Append question and answer to the list
                    qanda.append({"Question": qtxt, "Answer": answer, "Prompt": prompt})

            results_url = write_list_of_dicts_to_google_sheet(qanda, "outputdata.json")
            print(f"Answers saved to: {results_url}")

            
if __name__ == "__main__":
    main()
