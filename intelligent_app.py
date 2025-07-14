import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline
import warnings
import re

# Suppress warnings for a cleaner output
warnings.filterwarnings("ignore", "Some weights of the model checkpoint at distilbert-base-cased-distilled-squad were not used when initializing")

def setup_rag_pipeline():
    """
    Loads data, processes it, and sets up the RAG pipeline.
    """
    print("Loading and processing data...")
    try:
        df = pd.read_csv('heart_disease_uci.csv')
    except FileNotFoundError:
        print("Error: 'heart_disease_uci.csv' not found. Make sure it's in the same folder.")
        return None, None

    # Give all 16 columns meaningful names
    df.columns = [
        'age', 'sex', 'dataset', 'chest_pain_type', 'resting_blood_pressure',
        'cholesterol', 'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate',
        'exercise_angina', 'st_depression', 'st_slope', 'num_major_vessels',
        'thalassemia', 'num_original', 'target'
    ]
    # Clean the 'dataset' column for easier filtering
    df['dataset'] = df['dataset'].str.strip().str.lower()


    # Create a more structured description for better parsing
    def create_description(row):
        sex = "Male" if row['sex'] == 1 else "Female"
        target = "Heart Disease" if row['target'] == 1 else "No Heart Disease"
        return (f"Dataset: {row['dataset']}. Age: {row['age']}. Sex: {sex}. "
                f"Cholesterol: {row['cholesterol']}. Max Heart Rate: {row['max_heart_rate']}. "
                f"Diagnosis: {target}.")

    df['description'] = df.apply(create_description, axis=1)
    descriptions = df['description'].tolist()
    print("Data processing complete.")

    # Setup RAG components (Encoder and FAISS index) for fallback
    print("Encoding data for the RAG model...")
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    encoded_data = encoder.encode(descriptions)
    index = faiss.IndexIDMap(faiss.IndexFlatL2(encoded_data.shape[1]))
    index.add_with_ids(np.array(encoded_data, dtype=np.float32), np.arange(len(descriptions)))
    print("RAG index created.")

    print("Loading QA model...")
    qa_pipeline = pipeline('question-answering', model='distilbert-base-cased-distilled-squad')
    print("QA model loaded.")

    # Map query words to actual DataFrame column names
    COLUMN_MAP = {
        "cholesterol": "cholesterol",
        "cholestrol": "cholesterol",
        "age": "age",
        "resting blood pressure": "resting_blood_pressure",
        "max heart rate": "max_heart_rate"
    }

    def direct_lookup(question, local_df):
        """
        NEW: This function tries to answer questions about max/min values directly.
        """
        question = question.lower()
        # Check for location keywords
        for location in ['cleveland', 'hungary', 'switzerland', 'va']:
            if location in question:
                local_df = local_df[local_df['dataset'] == location]

        # Check for highest/lowest questions
        if "highest" in question or "max" in question:
            for keyword, col_name in COLUMN_MAP.items():
                if keyword in question:
                    if not local_df.empty:
                        # Find the row with the max value for the specified column
                        result_row = local_df.loc[local_df[col_name].idxmax()]
                        return f"The person with the highest {keyword} (value: {result_row[col_name]}) is a {result_row['age']}-year-old {result_row['sex_str']}.", result_row
            return None, None

        return None, None # Return None if it's not a direct lookup question

    def answer_question(question, full_df):
        """
        This function now includes the direct lookup logic.
        """
        # Add a temporary string 'sex' column for easier lookup answers
        full_df['sex_str'] = full_df['sex'].apply(lambda x: 'Male' if x == 1 else 'Female')

        # 1. Try DIRECT LOOKUP first for precise answers
        direct_answer, _ = direct_lookup(question, full_df)
        if direct_answer:
            return direct_answer

        # 2. If direct lookup fails, fall back to the AI (RAG) method
        question_embedding = encoder.encode([question])
        _, I = index.search(np.array(question_embedding, dtype=np.float32), k=1)
        context = descriptions[I[0][0]]
        result = qa_pipeline(question=question, context=context)

        if result['score'] < 0.1:
            return "I'm not sure I can answer that based on the patient data."
        return result['answer']

    return lambda q: answer_question(q, df)


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    qa_func = setup_rag_pipeline()

    if qa_func:
        print("\n✅ **Intelligent Healthcare QA System is Ready** ✅")
        print("Ask for specifics like 'highest cholesterol' or general questions.")
        print("Type 'quit' to exit.")

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