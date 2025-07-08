# Fake-News-Detector-App

Live Demo-https://fake-news-detector-app-qeynkwrlis7vbfyd3cyhdy.streamlit.app/
A Fake News Detector in Python uses NLP techniques and machine learning models like Logistic Regression or Naive Bayes to classify news articles as real or fake based on textual content.


````markdown
# 📰 Fake News Detection using Machine Learning

This project implements a Fake News Detector using Python, Scikit-learn, and Streamlit.
 It uses a Passive Aggressive Classifier trained on real and fake news datasets to classify news articles as **REAL** or **FAKE**.

---

## 📂 Dataset

The dataset includes two CSV files:

- `True.csv`: Articles labeled as **REAL**
- `Fake.csv`: Articles labeled as **FAKE**

Both datasets are merged and used to train the model.

---

## 📌 Features

- Text preprocessing using **TF-IDF Vectorization**
- Binary classification using **PassiveAggressiveClassifier**
- Accuracy evaluation of the model
- Model and vectorizer saved using **pickle**
- Streamlit-based web interface for user interaction

---

## 🛠️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/fake-news-detector.git
cd fake-news-detector
````

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model (optional if `model.pkl` is already provided)

```bash
python train.py
```

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

---

## 🧠 Model

* **TF-IDF Vectorizer**: Converts raw text into numerical features
* **Passive Aggressive Classifier**: Fast and efficient online learning algorithm suitable for large datasets

---

## 📊 Accuracy

The model achieved an accuracy of approximately:

```
> 93% on test data
```

---

## Example-

| Input Text                                       | Prediction  |
| ------------------------------------------------ | ----------- |
| "NASA just launched a new satellite into space." | ✅ Real News |
| "Aliens were spotted voting in the US election." | ❌ Fake News |

---

## 📁 Project Structure

```
├── True.csv
├── Fake.csv
├── train.py                # Model training script
├── app.py                  # Streamlit web app
├── model.pkl               # Trained classifier
├── vectorizer.pkl          # TF-IDF vectorizer
├── README.md
├── requirements.txt
```

---

## 💡 Future Improvements

* Add confusion matrix and F1-score
* Support news headline and title-based classification
* Add live news article scraping and classification
* Deploy to Heroku or Render

---

## 🤝 Contributions

Feel free to fork the repo, create a branch, and submit a pull request

## 📄 License

This project is licensed under the [MIT License](LICENSE).

### ✅ Also create `requirements.txt`:
pandas
scikit-learn
streamlit
