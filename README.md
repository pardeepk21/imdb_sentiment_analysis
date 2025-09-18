# 🎬 Sentiment Analysis on IMDB Reviews  

This project demonstrates **Sentiment Analysis** (classifying movie reviews as *positive* or *negative*) using **Python, Scikit-learn, and Machine Learning models**.  
It compares **Naive Bayes** and **Logistic Regression** on the popular **IMDB dataset**.  

---

## 📌 Features  
- 📂 Loads and preprocesses the IMDB movie reviews dataset (50,000 reviews)  
- 🧮 Converts raw text into numeric features using **TF-IDF vectorization**  
- 🤖 Trains and evaluates two models:  
  - 🟥 **Naive Bayes**  
  - 🟩 **Logistic Regression**  
- 📊 Generates metrics: **Accuracy, Precision, Recall, F1-score**  
- 📈 Visualizations:  
  - Distribution of predicted sentiments  
  - Actual vs Predicted comparison  
  - Confusion Matrix (heatmap)  
- 📝 Supports **custom review testing** (enter your own sentence to check sentiment)  

---
## 📂 Project Structure  
📦 sentiment-analysis

┣ 📜 sentiments_analysis.py # Main script

┣ 📜 IMDB Dataset.csv # Dataset (from Kaggle or other source)

┣ 📜 requirements.txt # Dependencies

┗ 📜 README.md # Documentation

## ⚙️ Installation & Setup  

1. **Clone the repository**  
    ```bash
    git clone https://github.com/your-username/sentiment-analysis.git
    cd sentiment-analysis

2. Create a virtual environment (recommended)
    ```bash
    python -m venv venv
    source venv/bin/activate   # Mac/Linux
    venv\Scripts\activate      # Windows

4. Install dependencies
    ```bash
   pip install -r requirements.txt

6. Run the script
   ```bash
    python sentiments_analysis.py

# 📊 Example Output
  ✅ Accuracy
    Naive Bayes: ~85%
    Logistic Regression: ~89–90%

# 📈 Visualizations
  Predicted Distribution (NB vs LR) → number of reviews predicted as positive/negative
  Actual vs Predicted → compares true sentiment distribution with model outputs
  Confusion Matrix → heatmap showing classification performance

# 📝 Custom Testing
    You can test your own reviews by adding them to the script:
    custom_reviews = [
    "This movie was absolutely fantastic, I loved it!",
    "The film was boring and way too long."
    ]
custom_vec = vectorizer.transform(custom_reviews)
print(lr_model.predict(custom_vec))

# 📦 Requirements
  Python 3.8+
  pandas
  numpy
  matplotlib
  scikit-learn
    
  Install everything via:
    pip install -r requirements.txt

#🚀 Future Improvements
Add deep learning models (LSTMs, Transformers like BERT)
Deploy as a Flask/FastAPI web app
Create an interactive Streamlit dashboard for real-time predictions

#👨‍💻 Author

Pardeep Kumar
🌐 GitHub: ardeepk21
