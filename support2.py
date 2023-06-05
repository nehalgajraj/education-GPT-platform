# Required Imports
import os
import pinecone
import tempfile
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQA
from langchain.agents import Tool, initialize_agent
import openai

from fpdf import FPDF

# Constants
OPENAI_API_KEY = '' # Key here
PINECONE_API_KEY = '' # Key here
INDEX_NAME = "education-index"
MODEL_NAME = "ada"
import os
openai.api_key = os.getenv('OPENAI_API_KEY')


 # Initialize Pinecone
pinecone.init(api_key=OPENAI_API_KEY, environment='asia-southeast1-gcp')

def create_zip_file(output_dir, lecture_files):
    with zipfile.ZipFile(output_dir, 'w') as zipf:
        for file in lecture_files:
            zipf.write(file)

def generate_personalized_lecture(row, lecture_notes):
    child_name = row['Name']
    child_attributes = row.to_dict()
    child_attributes.pop('Name')

    system_message = f"I am an AI designed to tailor educational material based on a student's specific characteristics."
    user_prompt = f"Here is a general lecture note: {lecture_notes}. Please adjust this lecture to be more appropriate for a student named {child_name}, who has the following attributes: {child_attributes}.  Consider only those attributes you think are relevant to the content of the lecture notes."

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": system_message,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        max_tokens=500,
        temperature=0.0
    )

    return response.choices[0].message['content']


def generate_personalized_lectures(df, lecture_notes):
    personalized_lectures = {}

    for index, row in df.iterrows():
        child_name = row['Name']
        child_attributes = row.to_dict()
        child_attributes.pop('Name')

        system_message = f"This is a lecture note: {lecture_notes}"
        user_prompt = f"Generate a personalized note for {child_name} who has the following attributes: {child_attributes}"

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": system_message,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            max_tokens=500,
            temperature=0.0
        )

        personalized_lectures[child_name] = response.choices[0].message['content']

    return personalized_lectures


def query_agent(query, llm_model="gpt-3.5-turbo"):
    # Initialize Pinecone
    pinecone.init(api_key=PINECONE_API_KEY, environment='asia-southeast1-gcp')


    # Set Up Memory

    conversational_memory = ConversationBufferWindowMemory(memory_key='chat_history', k=5, return_messages=True)

    # Set Up Chat Model
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model=llm_model, temperature=0.0)

    # Set the index 
    embed = OpenAIEmbeddings(model=MODEL_NAME, openai_api_key="OPENAI_API_KEY")

    index = Pinecone.from_existing_index(index_name="education-index", embedding=embed)
    # Set Up QA
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=index.as_retriever())

    # Set Up Tool
    tools = [Tool(name="Knowledge Base", func=qa.run, description="You use this tool when asked about acadmic paper.")]

    # Initialize Agent
    agent = initialize_agent(agent='chat-conversational-react-description', tools=tools, llm=llm, verbose=True, max_iterations=3, early_stopping_method="generate", memory=conversational_memory)

    # Query Agent
    return agent(query)



def index_pdf(pdf_data, index_name=INDEX_NAME):
    # Initialize Pinecone
    pinecone.init(api_key=PINECONE_API_KEY, environment='asia-southeast1-gcp')

    # Create a temporary file and write the pdf_data to it
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
        temp_pdf.write(pdf_data)
        temp_file_name = temp_pdf.name  # Get the name of the temporary file

    # Load PDF
    loader = PyPDFLoader(temp_file_name)
    pages = loader.load()

    # Split Pages into Chunks
    pages_chunks = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20).split_documents(pages)

    # Embed and Index
    embed = OpenAIEmbeddings(model=MODEL_NAME, openai_api_key=OPENAI_API_KEY)
    index = Pinecone.from_documents(pages_chunks, embed, index_name=index_name)

    return index


import os
import openai
import json
import zipfile
import streamlit as st
from typing import List
import PyPDF2
from io import BytesIO

# Set OpenAI API key
openai.api_key = 'your-api-key'

import PyPDF2
import PyPDF2

def read_pdf(file):
    """ Read the content of a PDF file. """
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page].extract_text()
    return text

def grade_assignment(student_assignment: str, teacher_reference: str, grading_parameters: list) -> dict:
    """
    Use OpenAI GPT-3.5-turbo to grade an assignment based on a teacher reference.
    """
    openai.api_key = 'OPENAI_API_KEY'

    # Prepare a custom prompt for the OpenAI model
    prompt = f"""
    Reference assignment:
    {teacher_reference}

    Student assignment:
    {student_assignment}
    Sturcture the Output:
    1. Overall Impression
    2. Feedback points
    3. Grade Score (range:1-10)
    """

    # Structure the system message to instruct the AI to provide a critical review based on the selected grading parameters
    system_message = f"As an AI teaching assistant, review the student assignment based on the reference assignment, focusing on the following parameters: {', '.join(grading_parameters)}."

    # Use the OpenAI Chat Completion API to generate a response
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": system_message,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        max_tokens=500,
        temperature=0.0
    )

    return response.choices[0].message['content']




def grade_assignments(student_assignments: List[str], teacher_reference: str, difficulty: int, zip_path: str):
    """
    Grade multiple student assignments and save the results in a zip file.
    """
    with zipfile.ZipFile(zip_path, 'w') as myzip:
        # Grade each student assignment
        for i, assignment in enumerate(student_assignments):
            # Grade the assignment
            grade_and_review = grade_assignment(assignment, teacher_reference, difficulty)
            
            # Create a PDF with the grade and review
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size = 15)
            pdf.cell(200, 10, txt = f"Student {i+1}", ln = True, align = 'C')
            pdf.multi_cell(200, 10, txt = grade_and_review)

            # Save the PDF to a file
            pdf_filename = f'student_{i+1}_grade.pdf'
            pdf.output(pdf_filename)

            # Add the PDF file to the zip
            myzip.write(pdf_filename)

            # Delete the PDF file
            os.remove(pdf_filename)
