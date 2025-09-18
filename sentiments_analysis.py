import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
#from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt


def clean_text(text):
    text = re.sub(r"<.*?>", "", text)   # remove HTML tags
    text = re.sub(r"[^a-zA-Z]", " ", text)  # keep only letters
    text = text.lower()  # lowercase
    return text

# Step 1: Load dataset
data = pd.read_csv("IMDB Dataset.csv")
print("Dataset shape:", data.shape)
print(data.head())

# Step 2: Split into training and test sets
data["review"] = data["review"].apply(clean_text)
X = data["review"]
y = data["sentiment"]
y_pos = y[y=="positive"]
y_neg = y[y == "negative"]

print("Number of positive reviews:", len(y_pos))
print("Number of negative reviews:", len(y_neg))

count_pos = len(y_pos)
count_neg = len(y_neg)

#print(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Convert text to numeric features (TF-IDF)
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 4: Train classifier
model = LogisticRegression(max_iter=100)
#model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Step 5: Predictions
y_pred = model.predict(X_test_vec)

# Step 6: Evaluation
acc_score = accuracy_score(y_test, y_pred)*100
print(f"Accuracy: {acc_score}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# Test on custom review
test_review = ["This movie was boring and too long."]
test_vec = vectorizer.transform(test_review)
print("Prediction:", model.predict(test_vec))

# Plot
labels = ["Positive", "Negative"]
counts = [count_pos, count_neg]
plt.bar(labels, counts, color=["green", "red"])
plt.title("Distribution of Sentiments in IMDB Dataset")
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")
plt.show()
'''
words, scores = zip(*top_pos)
plt.barh(words, scores)
plt.title("Top Positive Words")
plt.show()
'''