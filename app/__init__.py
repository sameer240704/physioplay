from flask import Flask
from config import Config


def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)


    @app.route('/')
    def home():
        return "Welcome to PhysioPlay!"
    
    # Initialize services
    from app.services.langchain_service import init_langchain
    from app.services.llama_service import init_llama
    
    init_langchain(app)
    init_llama(app)
    
    # Register blueprints
    from app.routes import main
    app.register_blueprint(main)
    
    return app