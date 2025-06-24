# Vaidsys-Twitter-sentiment-analysis
Project Overview:
This project demonstrates how natural language processing (NLP) and machine learning can be used to classify tweet sentiments. The Sentiment140 dataset contains 1.6 million tweets labeled as positive or negative. Using this data, we train a Logistic Regression classifier with TF-IDF features to predict sentiment from tweet text.

Key Objectives:
- Develop a sentiment analysis model for social media data.
- Classify public sentiment as positive, negative.

Key Task:
- Collect and preprocess the data ( consisting of 1.6 million entries).
- erform text tokenization and sentiment labeling.
- Build a sentiment analysis model using natural language processing.

📊 Dataset

- **Source**: [Sentiment140 on Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140)
- **Total Tweets**: 1,600,000
- **Classes**:
  - '0': Negative
  - '1': Positive
- **Used Fields**:
  - Sentiment label ('0' or '1')
  - Tweet text

Preprocessing
Steps followed before feeding data into the model:
- Remove Twitter handles ('@username')
- Remove URLs
- Lowercase the text
- Remove special characters, digits, and punctuation
- Tokenization (using 'nltk')
- Stopword removal
- Optional: Stemming using Porter Stemmer
- TF-IDF vectorization to convert text into numerical form

 Model Details

- **Model**: Logistic Regression
- **Feature Extraction**: 'TfidfVectorizer'
- **Train/Test Split**: 80/20 with 'stratify=y' to preserve class balance
- **Libraries Used**:  
  'scikit-learn', 'nltk', 'pandas', 'numpy', 'matplotlib', 'seaborn', 'kaggle','tqdm'

Potential Changes( to make prediction much better)

- Try to train other models too( SVC,RandomForest,XGBoost,Naive-Bayes)
- Perform hyperparameter tunning to get the optimumm parameter( GridsearchCV,RandomizedSerachedCV)
- Try to use some other word-->vectorize techniques( bow,n-grams,word2vec)

