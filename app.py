import pandas as pd

# Load the dataset
df = pd.read_csv('heart_disease_uci.csv')

import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline
import warnings

# Suppress a specific warning from the transformers library for a cleaner output
warnings.filterwarnings("ignore", "Some weights of the model checkpoint at distilbert-base-cased-distilled-squad were not used when initializing")

def setup_rag_pipeline():
    """
    This function loads data, processes it, and sets up the RAG pipeline.
    It's designed to be run only once at the start.
    """
    # 1. DATA ENGINEERING: Load and process the healthcare data
    print("Loading and processing data...")
    try:
        df = pd.read_csv('heart_disease_uci.csv')
    except FileNotFoundError:
        print("Error: 'heart_disease_uci.csv' not found. Make sure it's in the same folder.")
        return None

    # Give all 16 columns meaningful names to match the CSV file
    df.columns = [
        'age', 'sex', 'dataset', 'chest_pain_type', 'resting_blood_pressure',
        'cholesterol', 'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate',
        'exercise_angina', 'st_depression', 'st_slope', 'num_major_vessels',
        'thalassemia', 'num_original', 'target'
    ]

    # Create a text description for each row
    def create_description(row):
        sex = "Male" if row['sex'] == 1 else "Female"
        fbs = "Yes" if row['fasting_blood_sugar'] > 0 else "No"
        exang = "Yes" if row['exercise_angina'] == 1 else "No"
        target = "Heart Disease" if row['target'] == 1 else "No Heart Disease"
        return (
            f"Patient Age: {row['age']}, Sex: {sex}, Chest Pain Type: {row['chest_pain_type']}, "
            f"Resting BP: {row['resting_blood_pressure']}, Cholesterol: {row['cholesterol']}, "
            f"Fasting Blood Sugar > 120 mg/dl: {fbs}, Max Heart Rate: {row['max_heart_rate']}, "
            f"Exercise Induced Angina: {exang}. Diagnosis: {target}"
        )

    df['description'] = df.apply(create_description, axis=1)
    descriptions = df['description'].tolist()
    print("Data processing complete.")

    # 2. RAG SETUP: Create embeddings and the search index
    print("Encoding data for the RAG model... this may take a moment.")
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    encoded_data = encoder.encode(descriptions)

    index = faiss.IndexIDMap(faiss.IndexFlatL2(encoded_data.shape[1]))
    index.add_with_ids(np.array(encoded_data, dtype=np.float32), np.arange(len(descriptions)))
    print("RAG index created.")

    # 3. LLM SETUP: Load the question-answering model
    print("Loading QA model...")
    qa_pipeline = pipeline('question-answering', model='distilbert-base-cased-distilled-squad', tokenizer='distilbert-base-cased-distilled-squad')
    print("QA model loaded.")

    # 4. DEFINE THE CORE QA FUNCTION
    def answer_question(question):
        # The model finds the most relevant patient description, even with messy questions
        question_embedding = encoder.encode([question])
        _, I = index.search(np.array(question_embedding, dtype=np.float32), k=1)
        context = descriptions[I[0][0]]

        # The LLM then extracts the specific answer from that context
        result = qa_pipeline(question=question, context=context)
        
        # Add a check for confidence
        if result['score'] < 0.1: # You can adjust this threshold
            return "I'm not sure I can answer that based on the patient data."
        return result['answer']

    return answer_question

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Set up the entire pipeline once
    qa_func = setup_rag_pipeline()

    if qa_func:
        print("\n✅ **Real-time Healthcare QA System is Ready** ✅")
        print("Ask any question about the patient data. Type 'quit' to exit.")
        
        # This loop runs forever until you type 'quit'
        while True:
            user_question = input("\nYour Question: ")
            if user_question.lower() == 'quit':
                print("Exiting the system. Goodbye!")
                break
            
            try:
                answer = qa_func(user_question)
                print(f"💡 Answer: {answer}")
            except Exception as e:
                print(f"An error occurred: {e}")