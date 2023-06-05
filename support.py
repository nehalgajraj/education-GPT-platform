import streamlit as st
from support2 import index_pdf, query_agent, read_pdf, grade_assignments, create_zip_file, generate_personalized_lecture
import pinecone 
import tempfile
import os 
import os
import openai
import json
import zipfile
import streamlit as st
from typing import List
import PyPDF2
from io import BytesIO
OPENAI_API_KEY = '' #key 
INDEX_NAME = 'education-index'
MODEL_NAME = 'ada'
PINECONE_API_KEY = '' #key
openai.api_key = os.getenv('OPENAI_API_KEY')

# Academic Research component
def academic_research_component():
    st.title("Academic Research")
    st.write("Content for the Academic Research component goes here!")

    # A list to keep all responses
    responses = []

    # Get a query from the user
    query = st.text_input("Enter your query:")
    if query:
        # Initialize the query agent
        # Query the agent and get the response
        response = query_agent(query)
        # Extract the 'output' part of the response
        answer = response['output']

        # Add the new answer and the query at the beginning of the list
        responses.insert(0, ("User: " + query, "Bot: " + answer))

    # Display the list of responses
    for user_query, bot_response in responses:
        st.markdown(f"**{user_query}**")
        st.markdown(f"_{bot_response}_")


def classroom_overview_component():
    st.title("Dynamic Pop-up Example")

    if 'uploaded_csv' in st.session_state:
        df = st.session_state['uploaded_csv']

        # Create two container blocks
        button_container = st.container()
        spacer_container = st.container()  # Additional container for space
        popup_container = st.container()

        # Create four columns: one for space, and three for buttons
        space_column, button_column1, button_column2, button_column3 = button_container.columns([1,3,3,3])

        # Create a list of button columns to iterate over
        button_columns = [button_column1, button_column2, button_column3]

        # Iterate over DataFrame rows and columns
        for index, row in df.iterrows():
            # Determine the column for the button based on the row index
            current_column = button_columns[index % 3]

            with current_column:
                # Use the 'name' value of the row as the button name
                button_name = row['Name']
                if st.button(f"{button_name}"):
                    st.session_state[f'pop_up_{index + 1}_open'] = not st.session_state.get(f"pop_up_{index + 1}_open", False)

        with spacer_container:
            st.empty()  # Add an empty block for space

        # Create three columns in the popup_container for left space, pop-up content, and right space
        left_space, popup_content_column, right_space = popup_container.columns([1,5,1])

        # Pop-ups column
        with popup_content_column:
            for index, row in df.iterrows():
                if st.session_state.get(f"pop_up_{index + 1}_open"):
                    # Use Markdown to format the content
                    content = f"**Student {index + 1} informations:**\n\n"
                    content += "\n".join([f"- {k}: {v}" for k, v in row.items()])  # use items() instead of iteritems()

                    # Surround the content with a thick box outline
                    content = f"```\n{content}\n```"
                    st.markdown(content)

    else:
        st.warning("Please upload a CSV file in the Settings.")


def personalized_learning_component():
    openai.api_key = 'your-api-key'

    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file is not None:
        lecture_content = read_pdf(uploaded_file)

        # Use the CSV data from the session state
        df = st.session_state['uploaded_csv']

        # Progress bar
        progress_bar = st.progress(0)
        total = len(df)

        lecture_files = []
        for i, (index, row) in enumerate(df.iterrows()):
            personalized_lecture = generate_personalized_lecture(row, lecture_content)

            temp_file_path = f"{row['Name']}_lecture.txt"
            with open(temp_file_path, "w") as temp_file:
                temp_file.write(personalized_lecture)
            lecture_files.append(temp_file_path)

            # Update progress bar
            progress_bar.progress((i + 1) / total)

        output_dir = "personalized_lectures.zip"
        create_zip_file(output_dir, lecture_files)

        download_button = st.button("Download Processed Files")

        if download_button:
            with open(output_dir, "rb") as zip_file:
                st.download_button(
                    label="Download",
                    data=zip_file,
                    file_name="personalized_lectures.zip",
                    mime="application/zip"
                )

            for file in lecture_files:
                os.remove(file)
            os.remove(output_dir)



# Automatic Grading component
def automatic_grading_component():
    st.title("Automatic Grading Component")

    # File upload widgets
    col1, col2 = st.columns(2)
    with col1:
        file_1 = st.file_uploader("Upload Teacher's Reference File", type=["pdf"])
    with col2:
        files_2 = st.file_uploader("Upload Students' Assignments (pdf files)", type=["pdf"], accept_multiple_files=True)

    # Grading parameter selection
    grading_parameters = ["Content", "Clarity", "Originality", "Relevance", "Structure"]
    selected_parameters = st.multiselect("Select Grading Parameters", grading_parameters)

    # Process the files on button click
    if st.button("Grade"):
        # Check if both files are uploaded
        if file_1 and files_2:
            teacher_reference = read_pdf(BytesIO(file_1.getvalue()))

            # Read student assignments
            student_assignments = []
            for file in files_2:
                student_assignments.append(read_pdf(BytesIO(file.getvalue())))

            # Grade assignments
            zip_path = 'graded_assignments.zip'
            grade_assignments(student_assignments, teacher_reference, selected_parameters, zip_path)
            st.write("Grading complete!")

            # Download button
            with open(zip_path, 'rb') as f:
                bytes = f.read()
                st.download_button(
                    label="Download graded assignments",
                    data=bytes,
                    file_name=zip_path,
                    mime='application/zip'
                )
        else:
            st.warning("Please upload the files.")


# Student Report component

def student_report_component():

    # Set up columns
    input_col, output_col = st.columns(2)

    with input_col:

        # Take input from user as points
        user_prompt = st.text_area('Enter student points to be included in the report:', height=300)

        # Generate report button
        generate_report_button = st.button('Generate Report')

    if generate_report_button and user_prompt:

        system_message = "You are a virtual assistant tasked with creating a student report based on provided points."

        # Use OpenAI's ChatCompletion API to create a student report
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

        report = response.choices[0].message['content']

        with output_col:
            st.write(f'Student Report: {report}')
     

def settings_component():
    import pandas as pd
    import streamlit as st

    st.title("Settings")
    index_name = "education-index"
    
    uploaded_csv = st.file_uploader("Upload Student Database", type=['csv'])

    # Initialize uploaded_pdfs in session_state as an empty list if it doesn't exist
    if 'uploaded_pdfs' not in st.session_state:
        st.session_state['uploaded_pdfs'] = []

    # Allow the user to upload multiple files
    uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

    if uploaded_files:
        try:
            for uploaded_file in uploaded_files:
                pdf_data = uploaded_file.read()  # Read the PDF file
                st.session_state['uploaded_pdfs'].append(pdf_data)  # Append PDF data to uploaded_pdfs
            st.success("PDFs uploaded successfully!")
        except Exception as e:
            st.error(f"Error during PDF upload: {e}")


    # Index uploaded PDFs
    if st.button('Index Uploaded PDFs'):
        try:
            for pdf_data in st.session_state['uploaded_pdfs']:
                # Create a temporary file and write the pdf_data to it
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
                    temp_pdf.write(pdf_data)
                    temp_file_name = temp_pdf.name  # Get the name of the temporary file

                index = index_pdf(pdf_data, index_name)  # Index the PDF
            st.success('PDFs indexed successfully!')
        except Exception as e:
            st.error(f"Error during PDF indexing: {e}")

    if uploaded_csv:
        try:
            df = pd.read_csv(uploaded_csv)
            st.session_state['uploaded_csv'] = df
            st.write(f"You have uploaded a CSV file named {uploaded_csv.name}.")
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
    elif 'uploaded_csv' in st.session_state:
        st.write("You have previously uploaded a CSV file.")

    # Add a button to clear the Pinecone index
    if st.button('Clear Index'):
        try:
            # processor.clear_index(index_name)
            pinecone.deleteindex(index_name)
            st.text('Index cleared!')
        except Exception as e:
            st.error(f"Error during index deletion: {e}")

    # Load PDF documents
    if 'uploaded_pdfs' in st.session_state:
        st.write(f"Number of PDF documents: {len(st.session_state['uploaded_pdfs'])}")
