import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

MODEL_FILE = "spam_model.pkl"
VEC_FILE = "vectorizer.pkl"

# Training Data
emails = [
    # Spam emails (Class 1)
    "Win money now", 
    "Free offer just for you", 
    "Claim your prize now",
    "Congratulations! You've been selected for a $1000 gift card.",
    "URGENT: Your account has been compromised. Verify your identity.",
    "Exclusive deal: Buy one get one free!",
    "Limited time offer: Get 50% off your next purchase.",
    "You are a winner! Click to claim your reward.",
    "Make money fast working from home effortlessly.",
    "Meet hot singles in your area tonight.",
    
    # Genuine / Academic non-spam emails (Class 0)
    "Meeting at 10am to discuss the micro project",
    "Project discussion tomorrow in the campus library",
    "Let's study maths for the upcoming final exam",
    "Your university application status update has been posted.",
    "Important: Academic schedule for the upcoming Fall semester.",
    "Reminder: Submission deadline for your math assignment is tonight.",
    "Can we reschedule our 1:1 mentoring meeting to next Thursday?",
    "Here are the lecture notes from today's applied calculus class.",
    "Please review the attached project proposal before our sync.",
    "Your university library books are due next week."
]
labels = [1]*10 + [0]*10

def train_and_save():
    """Trains the model from scratch and saves it aggressively to disk."""
    print("Training new model and vectorizer...")
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(emails)
    
    model = MultinomialNB()
    model.fit(X, labels)
    
    # Output to disk using joblib
    joblib.dump(model, MODEL_FILE)
    joblib.dump(vectorizer, VEC_FILE)
    print("Model and vectorizer trained and saved successfully.")
    
    return model, vectorizer

# 1. LOAD MODEL PROPERLY
if os.path.exists(MODEL_FILE) and os.path.exists(VEC_FILE):
    try:
        model = joblib.load(MODEL_FILE)
        vectorizer = joblib.load(VEC_FILE)
        print("Existing model and vectorizer loaded successfully.")
    except Exception as e:
        print(f"Error loading model from disk: {e}. Retraining...")
        model, vectorizer = train_and_save()
else:
    model, vectorizer = train_and_save()


def predict_email(text):
    """
    Transforms the input text, utilizes the loaded model to predict the probabilities,
    and returns a clean JSON-ready dictionary.
    """
    try:
        # 4. DEBUG SAFETY: Print processed text
        print("\n--- PREDICTION LOG ---")
        print(f"Original text: {text}")
        
        # 1. Transform input text using vectorizer before prediction
        data = vectorizer.transform([text])
        
        # 2. USE predict_proba() instead of fallback static values
        probs = model.predict_proba(data)[0]
        safe_prob = float(probs[0])
        spam_prob = float(probs[1])
        
        # 4. DEBUG SAFETY: Print raw probabilities
        print(f"Raw probabilities -> Safe: {safe_prob:.4f} | Spam: {spam_prob:.4f}")
        
        # 3. RETURN CORRECT LABEL
        # Strictly checking if spam probability > 0.5 => 1 (Spam), else 0 (Safe)
        prediction = 1 if spam_prob > 0.5 else 0
        
        # 6. CLEAN OUTPUT FORMAT (and NO hardcoded arrays/defaults)
        return {
            "prediction": prediction,
            "spam_probability": spam_prob,
            "safe_probability": safe_prob
        }
        
    except Exception as e:
        # 5. HANDLE EDGE CASE: Return error gracefully if the model crashes
        print(f"Model prediction failed: {e}")
        return {
            "error": "Model prediction failed",
            "details": str(e)
        }