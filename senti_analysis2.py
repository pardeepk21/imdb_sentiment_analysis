import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import re

def clean_text(text):
    text = re.sub(r"<.*?>", "", text)   # remove HTML tags
    text = re.sub(r"[^a-zA-Z]", " ", text)  # keep only letters
    text = text.lower()  # lowercase
    return text


# Step 1: Load dataset
data = pd.read_csv("IMDB Dataset.csv")
print("Dataset shape:", data.shape)

# Step 2: Split into train/test
data["review"] = data["review"].apply(clean_text)
X = data["review"]
y = data["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words="english", max_features=10000, ngram_range=(1,2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 4: Train Naive Bayes model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Step 5: Evaluate
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 6: Extract top words for each class
feature_names = np.array(vectorizer.get_feature_names_out())
log_prob = model.feature_log_prob_

# Top Positive Words
pos_index = list(model.classes_).index("positive")
top_pos_idx = np.argsort(log_prob[pos_index])[-20:]
top_pos_words = feature_names[top_pos_idx]
top_pos_scores = log_prob[pos_index][top_pos_idx]

# Top Negative Words
neg_index = list(model.classes_).index("negative")
top_neg_idx = np.argsort(log_prob[neg_index])[-20:]
top_neg_words = feature_names[top_neg_idx]
top_neg_scores = log_prob[neg_index][top_neg_idx]

# Step 7: Plot Top Positive Words
plt.figure(figsize=(10,6))
plt.barh(top_pos_words, top_pos_scores, color="green")
plt.title("Top Positive Words (Naive Bayes)")
plt.xlabel("Log Probability")
plt.gca().invert_yaxis()
plt.show()

# Step 8: Plot Top Negative Words
plt.figure(figsize=(10,6))
plt.barh(top_neg_words, top_neg_scores, color="red")
plt.title("Top Negative Words (Naive Bayes)")
plt.xlabel("Log Probability")
plt.gca().invert_yaxis()
plt.show()

# Step 9: Test with your own reviews
custom_reviews = [
    "This movie was absolutely fantastic, I loved it!",
    "The film was boring and way too long.",
    "Mediocre plot but good acting.",
    "Worst movie I have seen this year."
]

custom_vec = vectorizer.transform(custom_reviews)
predictions = model.predict(custom_vec)

print("\nCustom Predictions:")
for r, p in zip(custom_reviews, predictions):
    print(f"{r} --> {p}")