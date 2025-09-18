import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Load dataset
data = pd.read_csv("IMDB Dataset.csv")
print("Dataset shape:", data.shape)

X = data["review"]
y = data["sentiment"]

# Step 2: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 3: TF-IDF Vectorization
vectorizer = TfidfVectorizer(
    stop_words="english", max_features=20000, ngram_range=(1, 2)
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 4: Train Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)
nb_pred = nb_model.predict(X_test_vec)
nb_acc = accuracy_score(y_test, nb_pred)

# Step 5: Train Logistic Regression
lr_model = LogisticRegression(max_iter=200)
lr_model.fit(X_train_vec, y_train)
lr_pred = lr_model.predict(X_test_vec)
lr_acc = accuracy_score(y_test, lr_pred)

print("Naive Bayes Accuracy:", nb_acc)
print("Logistic Regression Accuracy:", lr_acc)
print("\nNaive Bayes Report:\n", classification_report(y_test, nb_pred))
print("\nLogistic Regression Report:\n", classification_report(y_test, lr_pred))

# Step 6: Prediction Distributions
nb_unique, nb_counts = np.unique(nb_pred, return_counts=True)
lr_unique, lr_counts = np.unique(lr_pred, return_counts=True)

print("Naive Bayes Prediction Distribution:", dict(zip(nb_unique, nb_counts)))
print("Logistic Regression Prediction Distribution:", dict(zip(lr_unique, lr_counts)))

# Step 7: Plot Predicted Distributions
labels = ["Negative", "Positive"]
x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 6))
ax.bar(x - width / 2, nb_counts, width, label="Naive Bayes", color="red", alpha=0.6)
ax.bar(x + width / 2, lr_counts, width, label="Logistic Regression", color="green", alpha=0.6)

ax.set_ylabel("Number of Reviews")
ax.set_title("Predicted Sentiment Distribution")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
plt.show()

# Step 8: Compare Actual vs Predicted
actual_counts = y_test.value_counts()

fig, ax = plt.subplots(figsize=(8, 6))
ax.bar(labels, actual_counts, color="blue", alpha=0.5, label="Actual")
ax.bar(labels, lr_counts, color="green", alpha=0.5, label="Logistic Regression Predicted")
ax.bar(labels, nb_counts, color="red", alpha=0.5, label="Naive Bayes Predicted")

ax.set_ylabel("Number of Reviews")
ax.set_title("Actual vs Predicted Sentiments")
ax.legend()
plt.show()

# Step 9: Confusion Matrix (Logistic Regression Example)
cm = confusion_matrix(y_test, lr_pred, labels=["negative", "positive"])
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap="Blues")

# Labels
ax.set_xticks(np.arange(len(labels)))
ax.set_yticks(np.arange(len(labels)))
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Logistic Regression)")

# Show values inside the matrix
for i in range(len(labels)):
    for j in range(len(labels)):
        ax.text(j, i, cm[i, j], ha="center", va="center", color="black")

plt.colorbar(im)
plt.show()