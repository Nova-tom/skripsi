from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import json
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

app = Flask(__name__)

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Tokenization and remove punctuation
    tokens = [word.strip(string.punctuation) for word in text.split()]

    # Remove stopwords
    stop_words = set(stopwords.words('indonesian'))
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Reassemble the text
    processed_text = ' '.join(tokens)

    return processed_text

def load_data_from_json(data_file):
    with open(data_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def train_decision_tree_classifier(data):
    # Split the data into features (X) and labels (y)
    X = [preprocess_text(entry['text']) for entry in data]
    y = [entry['label'] for entry in data]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a pipeline with CountVectorizer and DecisionTreeClassifier
    model = make_pipeline(CountVectorizer(), DecisionTreeClassifier())

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the testing data
    y_pred = model.predict(X_test)

    # Evaluate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Decision Tree Classifier Accuracy: {accuracy:.2f}")

    # Print classification report (includes precision, recall, and F1-score)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    return model

def predicted_category(report, model):
    # Preprocess the input text before making predictions
    processed_report = preprocess_text(report)

    # Make a prediction using the trained model
    predicted_category = model.predict([processed_report])[0]

    return {"predicted_category": predicted_category}

@app.route('/predict_category', methods=['POST'])
def predict_category():
    try:
        # Get the JSON data from the POST request
        request_data = request.get_json()

        # Extract the report text from the JSON data
        report_text = request_data.get('report_text', '')

        # Get the predicted category for the report
        prediction_result = predicted_category(report_text, dt_model)

        return jsonify(prediction_result)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    # Specify your data file name
    data_file = 'data.json'

    # Load data from the specified file
    data = load_data_from_json(data_file)

    # Train the decision tree classifier
    dt_model = train_decision_tree_classifier(data)

    # Run the Flask app
    app.run(debug=True)
