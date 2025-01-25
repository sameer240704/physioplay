import random
import fitz  # PyMuPDF
from flask import Flask, Blueprint, jsonify, request, render_template
import logging
import os
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import Counter



current_pdf = {
    'filename': None,
    'filepath': None
}

# Initialize the QA pipeline
qa_model_name = "deepset/roberta-base-squad2"
qa_pipeline = pipeline('question-answering', model=qa_model_name)

# Initialize sentence transformer for text similarity
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Blueprint setup
main = Blueprint('main', __name__)

# Configure upload folder and allowed extensions
app.config['UPLOAD_FOLDER'] = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}

# Create the upload folder if it doesn't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Function to check allowed file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route for the index page
@main.route('/')
def index():
    return render_template('physio.html')

# Route for llama chat page
@main.route('/llama')
def llama_chat():
    return render_template('llama_chat.html')

# API endpoint to get case introduction with pain details from a random PDF
@main.route('/get_case_introduction/', methods=['GET'])
def get_case_introduction():
    try:
        # List all PDF files in the upload folder
        pdf_files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if allowed_file(f)]
        
        # Check if there are any PDF files in the directory
        if not pdf_files:
            logger.warning("No PDF files found in the upload folder")
            return jsonify({"error": "No PDF files available for processing"}), 400

        # Select a random PDF file
        filename = random.choice(pdf_files)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        logger.info(f"Randomly selected file: {filename}")

        # Extract content from the PDF using PyMuPDF (fitz)
        doc = fitz.open(filepath)

        # Extract the text from the first page (or other pages if needed)
        first_page = doc[0]
        text = first_page.get_text("text")  # Extract text in plain format

        # Try to extract relevant introduction by looking for pain-related keywords
        lines = text.split("\n")
        intro = ""
        for line in lines:
            if "pain" in line.lower() or "suffering" in line.lower() or "symptoms" in line.lower():
                intro = line
                break

        # If no pain-related information is found, return the first 2 lines as fallback
        if not intro:
            intro = " ".join(lines[:2])  # Join the first two lines as the introduction

        logger.info(f"Extracted introduction: {intro}")

        return jsonify({"message": "File processed successfully", "introduction": intro}), 200

    except Exception as e:
        logger.error("Error processing PDF", exc_info=True)
        return jsonify({"error": "Error during PDF processing"}), 500


# API endpoint to process PDF files
@main.route('/process_random_pdf/', methods=['GET'])
def process_random_pdf():
    try:
        # List all PDF files in the upload folder
        pdf_files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if allowed_file(f)]
        
        # Check if there are any PDF files in the directory
        if not pdf_files:
            logger.warning("No PDF files found in the upload folder")
            return jsonify({"error": "No PDF files available for processing"}), 400

        # Select a random PDF file and store it
        current_pdf['filename'] = random.choice(pdf_files)
        current_pdf['filepath'] = os.path.join(app.config['UPLOAD_FOLDER'], current_pdf['filename'])
        logger.info(f"Randomly selected file: {current_pdf['filename']}")

        # Simulated PDF processing logic
        try:
            processed_content = f"PDF {current_pdf['filename']} processed successfully with dummy content."
            logger.info(f"Processed content from {current_pdf['filename']}: {processed_content}")
            return jsonify({
                "message": "File processed successfully", 
                "content": processed_content,
                "selected_pdf": current_pdf['filename']
            }), 200
        except Exception as e:
            logger.error("Error processing PDF", exc_info=True)
            return jsonify({"error": "Error during PDF processing"}), 500
    except Exception as e:
        logger.error("Error selecting random PDF", exc_info=True)
        return jsonify({"error": "Error selecting random PDF"}), 500


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def split_text(text, chunk_size=500):
    """Split text into chunks of approximately equal size."""
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        current_length += len(word) + 1  # +1 for space
        if current_length > chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = len(word)
        else:
            current_chunk.append(word)
            
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def find_most_relevant_chunk(question, chunks, top_k=3):
    """Find the most relevant text chunks for the question using sentence embeddings."""
    # Create embeddings for the question and all chunks
    question_embedding = sentence_model.encode([question])[0]
    chunk_embeddings = sentence_model.encode(chunks)
    
    # Calculate similarities
    similarities = cosine_similarity([question_embedding], chunk_embeddings)[0]
    
    # Get indices of top k most similar chunks
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    # Combine the top chunks
    relevant_text = " ".join([chunks[i] for i in top_indices])
    
    return relevant_text

# Add this function to check if question is diagnosis-related
def is_diagnosis_related(question):
    """Check if the question is related to diagnosis"""
    diagnosis_keywords = [
        'diagnosis', 'diagnose', 'condition', 'assessment', 'prognosis',
        'what is wrong', 'what condition', 'what disease', 'medical condition',
        'what do they have', 'what does the patient have', 'what is the problem',
        'clinical impression', 'differential diagnosis', 'diagnostic', 'diagnosed with',
        'medical assessment', 'clinical diagnosis', 'pathology', 'diagnostic finding',
        'what might be the cause', 'what could be causing', 'what is causing',
        'diagnostic conclusion', 'medical evaluation', 'clinical condition',
        'what is the diagnosis', 'suspected condition', 'probable diagnosis'
    ]
    
    question = question.lower()
    return any(keyword.lower() in question for keyword in diagnosis_keywords)

# Modified ask_question endpoint
@main.route('/ask_question', methods=['POST'])
@main.route('/ask_question/', methods=['POST'])
def ask_question():
    try:
        # Check if a PDF has been selected
        if not current_pdf['filename'] or not current_pdf['filepath']:
            logger.error("No PDF has been selected. Please call /process_random_pdf/ first")
            return jsonify({
                "error": "No PDF selected", 
                "message": "Please call /process_random_pdf/ endpoint first to select a PDF"
            }), 400

        # Get the question from the POST request
        data = request.get_json()
        if not data:
            logger.error("No JSON data received")
            return jsonify({"error": "No JSON data provided"}), 400
            
        if 'question' not in data:
            logger.error("No question field in JSON data")
            return jsonify({"error": "No question provided in JSON"}), 400
        
        question = data['question']
        logger.info(f"Received question: {question}")
        
        # Check if question is diagnosis-related
        if is_diagnosis_related(question):
            logger.warning("Diagnosis-related question detected")
            return jsonify({
                "error": "Restricted question",
                "message": "I apologize, but I cannot provide answers to questions related to diagnosis or medical conditions. Please consult with a qualified healthcare professional for diagnostic information.",
                "question_type": "diagnosis-related"
            }), 403
        
        # Extract text from the selected PDF
        raw_text = extract_text_from_pdf(current_pdf['filepath'])
        logger.info("Successfully extracted text from PDF")
        
        # Split text into chunks
        chunks = split_text(raw_text)
        logger.info(f"Split text into {len(chunks)} chunks")
        
        # Find most relevant chunks for the question
        relevant_text = find_most_relevant_chunk(question, chunks)
        logger.info("Found relevant text chunks")
        
        # Get answer using the QA pipeline
        result = qa_pipeline(question=question, context=relevant_text)
        logger.info(f"Generated answer with confidence score: {result['score']}")
        
        response = {
            "filename": current_pdf['filename'],
            "question": question,
            "answer": result['answer'],
            "confidence": float(result['score'])
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}", exc_info=True)
        return jsonify({
            "error": "Error processing question",
            "details": str(e)
        }), 500
    

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def normalize_medical_terms(text):
    """Normalize medical terms for better similarity comparison."""
    text = text.lower().strip()
    general_patterns = {
        'itis': ['inflammation'], 'opathy': ['disease'], 'osis': ['condition'], 
        'ectomy': ['removal'], 'plasty': ['repair'], 'acute': ['acute onset'], 
        'chronic': ['chronic onset'], 'bilateral': ['bilat'], 
        'anterior': ['ant'], 'posterior': ['post'], 'superior': ['sup'], 
        'inferior': ['inf'], 'medial': ['med'], 'lateral': ['lat'],
        'fracture': ['fx'], 'pain': ['algia']
    }
    for standard, variants in general_patterns.items():
        for variant in variants:
            text = text.replace(variant, standard)
    text = re.sub(r'[^\w\s-]', '', text)  # Remove non-alphanumeric except hyphens
    return ' '.join(text.split())  # Remove extra whitespace

def calculate_diagnosis_similarity(suggested_diagnosis, actual_diagnosis):
    """Enhanced similarity calculation with term and semantic analysis."""
    normalized_suggested = normalize_medical_terms(suggested_diagnosis)
    normalized_actual = normalize_medical_terms(actual_diagnosis)
    suggested_terms = Counter(normalized_suggested.split())
    actual_terms = Counter(normalized_actual.split())

    # Weighted term overlap
    term_overlap = sum(min(suggested_terms[word], actual_terms[word]) for word in suggested_terms)
    total_terms = sum(suggested_terms.values()) + sum(actual_terms.values())
    term_similarity = 0.7 * (term_overlap / max(total_terms, 1))

    # Semantic similarity
    suggested_embedding = sentence_model.encode([normalized_suggested])
    actual_embedding = sentence_model.encode([normalized_actual])
    semantic_similarity = cosine_similarity(suggested_embedding, actual_embedding)[0][0]

    # Adjust weight for semantic similarity
    final_similarity = 0.6 * term_similarity + 0.4 * semantic_similarity

    # Short input adjustment
    length_ratio = len(normalized_suggested.split()) / max(len(normalized_actual.split()), 1)
    adjusted_similarity = final_similarity * (0.7 + 0.3 * length_ratio)

    return min(1.0, max(0.0, adjusted_similarity))

def extract_diagnosis_from_pdf(pdf_path):
    """Extract diagnosis sections based on keywords from the PDF."""
    try:
        doc = fitz.open(pdf_path)
        diagnosis_text = ""
        section_markers = ["diagnosis", "diagnostic considerations", "primary consideration"]
        end_markers = ["management", "plan", "treatment", "recommendations"]

        for page in doc:
            text = page.get_text()
            for marker in section_markers:
                start_idx = text.lower().find(marker)
                if start_idx != -1:
                    start_idx += len(marker)
                    end_idx = len(text)
                    for end_marker in end_markers:
                        temp_idx = text.lower().find(end_marker, start_idx)
                        if temp_idx != -1:
                            end_idx = min(end_idx, temp_idx)
                    diagnosis_text += text[start_idx:end_idx].strip() + " "
        return re.sub(r'\s+', ' ', diagnosis_text.strip())
    except Exception as e:
        logger.error(f"Error extracting diagnosis: {str(e)}", exc_info=True)
        return ""

# Remove all Blueprint-related route decorators
# Replace with a direct Flask route
def validate_diagnosis():
    try:
        print("Validate Diagnosis Route Hit!")  # Debug print
        
        if not current_pdf['filename'] or not current_pdf['filepath']:
            print("No PDF selected!")
            return jsonify({"error": "No PDF selected"}), 400

        data = request.get_json()
        print(f"Received data: {data}")  # Print received data
        
        if not data or 'diagnosis' not in data:
            return jsonify({"error": "Missing diagnosis"}), 400

        suggested_diagnosis = data['diagnosis']
        actual_diagnosis = extract_diagnosis_from_pdf(current_pdf['filepath'])

        if not actual_diagnosis:
            return jsonify({"error": "No diagnosis found in the PDF"}), 500

        similarity_score = calculate_diagnosis_similarity(suggested_diagnosis, actual_diagnosis)
        verdict = "CORRECT" if similarity_score >= 0.8 else "WRONG"
        feedback = (
            "Your diagnosis appears to be correct." 
            if verdict == "CORRECT" else "Your diagnosis appears to be incorrect."
        )

        return jsonify({
            "verdict": verdict,
            "similarity_score": round(similarity_score, 2),
            "feedback": feedback,
            "actual_diagnosis": actual_diagnosis,
            "normalized_suggested": normalize_medical_terms(suggested_diagnosis),
            "normalized_actual": normalize_medical_terms(actual_diagnosis),
        }), 200
    except Exception as e:
        print(f"Error in validate_diagnosis: {str(e)}")
        return jsonify({"error": "Error validating diagnosis", "details": str(e)}), 500

# Explicitly add the route
app.add_url_rule('/validate_diagnosis', 'validate_diagnosis', validate_diagnosis, methods=['POST'])

# Register the blueprint
# Run the app
if __name__ == '__main__':
    app.run(debug=True, port=8000)

