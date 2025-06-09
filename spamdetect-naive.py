from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Sample messages
X = [
    "Congratulations, you won a prize",
    "Free entry in 2 a weekly competition",
    "Hi, how are you?",
    "Let’s catch up tomorrow",
    "Win money now",
    "Hello friend, are you coming today?"
]
y = ["spam", "spam", "ham", "ham", "spam", "ham"]

# Convert text to feature vectors
vectorizer = CountVectorizer()
X_features = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.3, random_state=42)

# Train Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
