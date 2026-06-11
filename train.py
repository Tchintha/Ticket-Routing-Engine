import logging
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def generate_synthetic_data():
    """Generates a highly representative synthetic dataset for B2B support tickets."""
    data = [
        {"text": "Cannot access my account after password reset", "category": "Account", "priority": "High"},
        {"text": "How to update my billing address and payment method?", "category": "Billing", "priority": "Medium"},
        {"text": "The API returns a 500 error when fetching reports", "category": "Technical", "priority": "High"},
        {"text": "Where can I find the documentation for SSO?", "category": "General", "priority": "Low"},
        {"text": "Locked out of my dashboard due to 2FA issues", "category": "Account", "priority": "High"},
        {"text": "Invoice #12345 seems to have an incorrect tax calculation", "category": "Billing", "priority": "Medium"},
        {"text": "Application crashes on startup after the latest update", "category": "Technical", "priority": "High"},
        {"text": "What is the pricing for the Enterprise plan?", "category": "General", "priority": "Low"},
        {"text": "I need to add a new administrator to our organization", "category": "Account", "priority": "Medium"},
        {"text": "Our subscription was canceled unexpectedly", "category": "Billing", "priority": "High"},
        {"text": "Database connection timeout in the production environment", "category": "Technical", "priority": "High"},
        {"text": "How do I export my data to a CSV file?", "category": "General", "priority": "Medium"},
        {"text": "Requesting a demo for the new features", "category": "General", "priority": "Low"},
        {"text": "Slow performance observed in the analytics module", "category": "Technical", "priority": "Medium"},
        {"text": "Need clarification on the refund policy", "category": "Billing", "priority": "Low"},
        {"text": "Resetting the workspace settings for our team", "category": "Account", "priority": "Medium"},
        {"text": "Integrating the webhook system with our CRM", "category": "Technical", "priority": "Medium"},
        {"text": "General feedback regarding the user interface", "category": "General", "priority": "Low"}
    ]
    return pd.DataFrame(data)

def train_and_serialize():
    """Builds, trains, and serializes categorization and priority pipelines."""
    logger.info("Generating synthetic training data...")
    df = generate_synthetic_data()

    # Define the common vectorizer
    # Using ngram_range=(1,2) and stop_words='english' as requested
    vectorizer_params = {
        'ngram_range': (1, 2),
        'stop_words': 'english'
    }

    # Pipeline for Category Classification
    category_pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(**vectorizer_params)),
        ('classifier', LogisticRegression(class_weight='balanced', random_state=42))
    ])

    # Pipeline for Priority Classification
    priority_pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(**vectorizer_params)),
        ('classifier', LogisticRegression(class_weight='balanced', random_state=42))
    ])

    logger.info("Training Category classification pipeline...")
    category_pipeline.fit(df['text'], df['category'])

    logger.info("Training Priority classification pipeline...")
    priority_pipeline.fit(df['text'], df['priority'])

    # Bundle models into a single artifact
    artifact = {
        "category_model": category_pipeline,
        "priority_model": priority_pipeline
    }

    model_filename = "routing_models.joblib"
    logger.info(f"Serializing models to {model_filename}...")
    joblib.dump(artifact, model_filename)
    logger.info("Training and serialization complete.")

if __name__ == "__main__":
    train_and_serialize()
