from flask import Flask
from config import Config


def create_app(config_class=Config):
    application = Flask(__name__)
    application.config.from_object(config_class)


    @application.route('/')
    def home():
        return "Welcome to PhysioPlay!"
    
    # Initialize services
    from app.services.langchain_service import init_langchain
    from app.services.llama_service import init_llama
    
    init_langchain(application)
    init_llama(application)
    
    # Register blueprints
    from app.routes import main
    application.register_blueprint(main)
    
    return application