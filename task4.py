from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Sample data (training)
messages = [
    "Win a free lottery now",     # spam
    "Lowest price offer",         # spam
    "Hello, how are you?",        # not spam
    "Let's meet tomorrow"         # not spam
]
labels = [1, 1, 0, 0]  # 1 = spam, 0 = not spam

# Convert text to vectors
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(messages)

# Train the model
model = MultinomialNB()
model.fit(X, labels)

# New message for prediction
test_message = ["Get a free iPhone now"]
test_vector = vectorizer.transform(test_message)

# Predict and print result
result = model.predict(test_vector)
print("Spam" if result[0] == 1 else "Not Spam")
