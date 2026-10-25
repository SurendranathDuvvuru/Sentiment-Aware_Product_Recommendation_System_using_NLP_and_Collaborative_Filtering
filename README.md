# Sentiment-Based Product Recommendation System

A machine learning capstone project that combines **customer review sentiment classification** with **collaborative filtering** to recommend products that a user is likely to appreciate.

## Table of Contents

- [Project Overview](#project-overview)
- [Objectives](#objectives)
- [Solution Workflow](#solution-workflow)
- [Models and Techniques](#models-and-techniques)
  - [Sentiment Classification](#sentiment-classification)
  - [Recommendation Engine](#recommendation-engine)
- [Technology Stack](#technology-stack)
- [Repository Contents](#repository-contents)
- [Expected Data](#expected-data)
- [Inference Examples](#inference-examples)
- [Evaluation](#evaluation)
- [Key Learnings](#key-learnings)
- [Limitations and Future Improvements](#limitations-and-future-improvements)
- [Disclaimer](#disclaimer)

## Project Overview

E-commerce platforms collect large volumes of product ratings and written reviews. This project uses review text to classify customer sentiment and combines that signal with user–product rating patterns to generate personalized product recommendations. `sample30.csv` dataset has been used.

## Objectives

- Explore and clean product review and rating data.
- Prepare customer review text for natural language processing (NLP).
- Train and evaluate sentiment classification models.
- Build user-based and item-based collaborative filtering recommenders.
- Combine recommendations with sentiment predictions to return a final shortlist of products.

## Solution Workflow

1. **Load the data** — read `sample30.csv` and inspect its columns, data types, and missing values.
2. **Explore and clean** — analyze review ratings and sentiment distribution, handle missing values, and reconcile sentiment labels with review ratings.
3. **Preprocess review text** — lowercase and clean text, remove punctuation and tokens containing numbers, remove English stop words, tokenize, and lemmatize.
4. **Extract features** — use TF-IDF with unigrams and bigrams, limited to 650 features.
5. **Address class imbalance** — apply SMOTE to the training data.
6. **Train sentiment models** — compare Logistic Regression, Multinomial Naive Bayes, Decision Tree, Random Forest, and XGBoost.
7. **Evaluate models** — review accuracy, precision, recall, F1-score, ROC-AUC, classification reports, and confusion matrices.
8. **Build recommenders** — implement user-user and item-item collaborative filtering using customer ratings.
9. **Evaluate recommendations** — compare the recommenders using RMSE.
10. **Generate final recommendations** — retrieve candidate products for a user and use the sentiment classifier to filter/rank the shortlist.

## Models and Techniques

### Sentiment classification

| Model | Purpose |
|---|---|
| Logistic Regression | Linear baseline for text classification |
| Multinomial Naive Bayes | Probabilistic text-classification baseline |
| Decision Tree | Non-linear classification baseline |
| Random Forest | Ensemble classification model |
| XGBoost | Boosted-tree classifier; selected in the notebook for saving and inference |

### Recommendation engine

- **User-user collaborative filtering:** recommends products based on rating patterns from similar users.
- **Item-item collaborative filtering:** recommends products based on relationships between products in users’ rating histories.
- **Sentiment-based filtering:** scores candidate product reviews with the sentiment classifier and returns a final shortlist.

The comparison reports that the user-based recommender has the lower RMSE (approximately 2).

## Technology Stack

- **Language:** Python
- **Data analysis:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn, WordCloud
- **NLP:** NLTK
- **Feature extraction:** scikit-learn `TfidfVectorizer`
- **Machine learning:** scikit-learn, imbalanced-learn (SMOTE), XGBoost
- **Recommendation:** user-user and item-item collaborative filtering
- **Model persistence:** Python Pickle
- **Environment:** Jupyter Notebook

## Repository Contents

```text
.
├── Capstone_Project_SentimentBasedProductRecommendation.ipynb
├── sample30.csv
├── pickle/
│   └── sentiment-classification-xg-boost-model.pkl
└── README.md
```

## Expected Data

Fields included:

- `id` — product identifier
- `name` — product name
- `reviews_rating` — customer rating
- `reviews_text` / `reviews_full_text` — written review text
- `reviews_username` — customer identifier
- `user_sentiment` — sentiment label
- `reviews_title`, `brand`, and `manufacturer` — additional review/product information used during exploration

## Inference Examples

Demonstrated sentiment inference on sample review text after applying the fitted TF-IDF vectorizer:

```python
X_sample = tfidf_vectorizer.transform(
    ["Awesome product, will recommend"]
)
prediction = xgclf.predict(X_sample)
print(prediction)
```
Recommendation function applied for a known user:

```python
get_sentiment_recommendations("08dallas")[:5]
```

## Evaluation

Evaluated the sentiment models using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix and classification report

The collaborative filtering approaches are evaluated using **Root Mean Squared Error (RMSE)**.

## Key Learnings

- Review text can provide useful signals beyond star ratings.
- Text normalization and TF-IDF feature extraction convert unstructured reviews into features for machine learning.
- Imbalanced labels can affect minority-class performance; resampling and class-aware evaluation are important.
- Collaborative filtering captures user–product preference patterns.
- Combining collaborative filtering with predicted sentiment can help refine a recommendation shortlist.

## Limitations and Future Improvements

- Fit text vectorization and resampling only on the training split to avoid data leakage; use a pipeline for reproducible evaluation.
- Validate sentiment labels carefully rather than assuming rating thresholds always represent true sentiment.
- Use cross-validation and hyperparameter tuning with a held-out test set.
- Evaluate recommendation quality with ranking metrics such as Precision@K, Recall@K, MAP, or NDCG in addition to RMSE.
- Handle new users and products explicitly with a cold-start strategy.
- Persist the vectorizer and all required preprocessing artifacts alongside the classifier.
- Wrap inference in a reusable module or API and add tests, logging, and model-version tracking.

## Disclaimer

This repository describes an educational capstone project. Recommendations and sentiment predictions are experimental and should be validated before use in a live e-commerce system.
