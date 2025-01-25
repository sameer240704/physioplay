import os

class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "default-secret-key")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    REPLICATE_API_KEY = os.getenv("REPLICATE_API_KEY")
    
    # Define the PDF folder path relative to the project root
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    PDF_FOLDER = os.path.join(BASE_DIR, 'data')
    
    # Create data directory if it doesn't exist
    if not os.path.exists(PDF_FOLDER):
        os.makedirs(PDF_FOLDER)

