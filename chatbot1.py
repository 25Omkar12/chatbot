import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import shutil
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from download import download_file

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Set path to your PDF file
#PDF_PATH = "data/data.pdf"

def get_pdf_text(pdf_path):
    text = ""
    pdf_reader = PdfReader(pdf_path)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True)

    return response["output_text"]

# Streamed response emulator
def response_generator(user_question):
    response = user_input(user_question)
    return response

# Function to download file from Google Drive
def download_pdf_from_drive(file_id, credentials_file):
    # Empty the folder first
    folder = os.getenv("DOWNLOAD_FOLDER")
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)
    
    # Download the file
    download_file(file_id, credentials_file)

# Streamlit app
def main():
    st.set_page_config(page_title="Chatbot", page_icon=r"logo.png", layout="centered", initial_sidebar_state="auto", menu_items=None)
    st.header("ChatBot",divider='rainbow', help = "This bot is designed by Omkar to address all of your questions hehe")
    st.subheader("Hello! There, How can I help you Today- 👩‍💻")
    
    # Input from user
    user_input = st.text_input("Please Paste the URL of the PDF here")

    if user_input:
        # Split the link by '/'
        parts = user_input.split('/')
        # Find the index of 'd' in the parts list
        index = parts.index('d')
        # The file ID is the next element after 'd'
        file_id = parts[index + 1]
        # Replace with the file ID, path to the service account credentials JSON file, and directory path
        credentials_file = os.getenv("CREDENTIALS_FILE")
        
        # Download the file
        if st.button("Download"):
            download_pdf_from_drive(file_id, credentials_file)

            # Process the PDF file automatically after downloading
            pdf_folder = os.getenv("DOWNLOAD_FOLDER")
            pdf_file_name = "data.pdf"
            PDF_PATH = os.path.join(pdf_folder, pdf_file_name)
            pdf_text = get_pdf_text(PDF_PATH)
            text_chunks = get_text_chunks(pdf_text)
            get_vector_store(text_chunks)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input(" Ask your Question here "):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Display assistant response in chat message container
        with st.spinner(text='Processing...'):
            response = response_generator(prompt)
            st.write(response, unsafe_allow_html=True)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
