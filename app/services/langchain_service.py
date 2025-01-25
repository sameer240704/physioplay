from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from app.utils.pdf_processor import process_pdf, select_random_pdf  # Add this import
import time
from flask import current_app
import os

# Global variables
vectors = None
diagnosis_keywords = [
    "diagnosis", "condition", "what do i have", "what's wrong", "what is wrong",
    "what could it be", "what is it", "what's causing", "what is causing",
    "why do i feel", "reason for", "explanation for", "what's the problem",
    "what is the problem", "what might be wrong", "possible cause"
]

def init_langchain(app):
    global vectors
    with app.app_context():
        # Create data directory if it doesn't exist
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            print(f"Created data directory at: {data_dir}")
        
        # Use PDF_FOLDER from config
        pdf_path = select_random_pdf(current_app.config['PDF_FOLDER'])
        if pdf_path:
            vectors = process_pdf(pdf_path)
            print(f"Processed PDF: {pdf_path}")
        else:
            print("No PDF files found in the data directory")

def get_chatgroq_response(user_input, is_introduction=False, is_diagnosis=False):
    llm = ChatGroq(
        groq_api_key=current_app.config['GROQ_API_KEY'],
        model_name="mixtral-8x7b-32768"
    )

    if is_introduction:
        prompt = ChatPromptTemplate.from_template("""
            Based on the provided context, generate a one-line introduction about yourself as the patient described in the physiotherapy case study.
            Use first-person perspective. Include only your name and your primary complaint or condition.
            Be very concise and disclose minimal information. Do not mention any specific diagnosis.
            <context>
            {context}
            </context>
            """)
    elif is_diagnosis:
        prompt = ChatPromptTemplate.from_template("""
            Based on the provided context, what is the correct diagnosis for this case?
            Provide only the diagnosis name without any explanation.
            <context>
            {context}
            </context>
            """)
    else:
        if any(keyword in user_input.lower() for keyword in diagnosis_keywords):
            return ("I'm not sure about the diagnosis. That's why I'm here to see a physiotherapist. "
                   "Could you please explain what you think based on what I've told you about my symptoms?"), 0

        prompt = ChatPromptTemplate.from_template("""
            You are the patient described in the physiotherapy case study.
            Answer the question from your perspective, using first-person language.
            Provide a concise response in one or two sentences.
            If the exact information is not available, use the context to provide a plausible answer based on your condition and experiences.
            Important: Do not mention or reveal any specific diagnosis in your response, even if it's mentioned in the context.
            <context>
            {context}
            </context>
            Question: {input}
            """)

    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    start = time.process_time()
    response = retrieval_chain.invoke({"input": user_input})
    end = time.process_time()

    return response['answer'], end - start

def process_physio_message(message, is_introduction=False, is_diagnosis=False):
    return get_chatgroq_response(message, is_introduction, is_diagnosis)