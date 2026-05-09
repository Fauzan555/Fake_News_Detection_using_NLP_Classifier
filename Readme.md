# Fake News Detection Using Machine Learning and NLP

# Project Overview

The objective of this project was to build an intelligent Natural Language Processing (NLP) based system capable of automatically classifying news headlines as either:
- Real News
- Fake News

The project focused on developing a complete end-to-end classical NLP pipeline involving:
- data preprocessing,
- text cleaning,
- feature engineering,
- machine learning model training,
- imbalance handling,
- comparative classifier benchmarking,
- and performance evaluation.

The primary goal was to analyze how different machine learning algorithms perform on sparse textual representations generated using TF-IDF vectorization.

---

# Dataset Used

The project used the FakeNewsNet Dataset, which contains online news headlines along with labels indicating whether the news is genuine or fake.

The dataset initially contained the following columns:
- title
- news_url
- source_domain
- tweet_num
- real

Since the objective was headline classification, only:
- title
- real

were selected for modeling.

The columns were then renamed as:
- text → input feature
- label → target variable

---

# Initial Data Analysis

The dataset contained:
- 23,196 news samples

Class distribution:
- Real News: 17,441
- Fake News: 5,755

This revealed a clear class imbalance problem where genuine news heavily dominated fake news samples.

Missing-value analysis showed:
- some missing values in URLs and source domains,
- but no missing values in headline text or labels.

Since URLs and domains were not required for textual headline classification, the project proceeded using headline text only.

---

# Text Preprocessing and Cleaning

Since machine learning models cannot directly process raw textual data, a text-cleaning pipeline was implemented.

The preprocessing steps included:

## 1. Lowercasing

All text was converted to lowercase to maintain uniformity.

Example:

```text
Breaking News → breaking news
```

---

## 2. Removing Special Characters

Regular expressions were used to remove:
- punctuation,
- symbols,
- non-alphanumeric characters.

---

## 3. Removing Numbers

Numerical digits were removed because they generally contribute less to semantic headline understanding.

---

## 4. Removing Extra Spaces

Whitespace normalization was performed to clean irregular spacing.

---

# Feature Engineering using TF-IDF

After preprocessing, textual headlines were converted into numerical representations using TF-IDF Vectorization.

TF-IDF stands for:
- Term Frequency
- Inverse Document Frequency

The idea behind TF-IDF is:
- words frequently occurring within a document are important,
- but words occurring across every document are less informative.

The TF-IDF score is computed as:

```text
TFIDF(t,d) = TF(t,d) × IDF(t)
```

Where:
- TF measures term frequency within a document,
- IDF measures rarity across all documents.

---

# TF-IDF Configuration

The vectorizer was configured using:

```python
TfidfVectorizer(
    stop_words="english",
    max_features=5000,
    ngram_range=(1,2)
)
```

## Important Configurations

### Stopword Removal

Common English stopwords such as:
- the
- is
- and

were removed to reduce noise.

---

### Maximum Features = 5000

Only the top 5000 most informative features were retained to:
- reduce dimensionality,
- improve efficiency,
- minimize overfitting.

---

### N-gram Range = (1,2)

Both:
- unigrams
- bigrams

were used.

This helped the model capture:
- individual words,
- phrase-level contextual patterns.

Examples:
- fake news
- breaking news
- white house

---

# Train-Test Splitting

The dataset was split into:
- 80% training data
- 20% testing data

using `train_test_split()` with:
- stratification,
- fixed random state,
- reproducibility controls.

Stratified sampling ensured that:
- both training and testing sets preserved the original class distribution.

This is particularly important in imbalanced classification tasks.

---

# Machine Learning Models Implemented

The project explored multiple machine learning classifiers to compare performance across different learning paradigms.

---

# 1. Logistic Regression

Initially, a standard Logistic Regression classifier was trained using TF-IDF vectors.

Logistic Regression was selected because:
- it performs efficiently on sparse high-dimensional text data,
- provides interpretable linear decision boundaries,
- serves as a strong NLP baseline model.

However, initial evaluation revealed:
- strong majority-class bias,
- poor fake-news recall.

---

# Handling Class Imbalance

To address imbalance, cost-sensitive learning was introduced using:

```python
class_weight="balanced"
```

This increased the penalty for minority fake-news misclassification.

As a result:
- fake-news recall improved significantly,
- false negatives reduced substantially,
- classification became more balanced.

---

# Comparative Model Benchmarking

To conduct a broader experimental study, multiple classifiers were implemented and evaluated.

The classifiers included:

| Model | Learning Paradigm |
|---|---|
| Logistic Regression | Linear Probabilistic |
| Multinomial Naive Bayes | Probabilistic |
| LinearSVC | Margin-Based |
| Passive Aggressive Classifier | Online Learning |

---

# 2. Multinomial Naive Bayes

MultinomialNB performed exceptionally well because:
- TF-IDF vectors are sparse,
- fake-news datasets often contain highly discriminative lexical distributions.

This model achieved:
- highest overall accuracy,
- highest recall,
- highest F1-score.

---

# 3. LinearSVC

Linear Support Vector Classification was implemented to leverage maximum margin optimization.

LinearSVC demonstrated:
- strong precision,
- robust high-dimensional classification,
- reliable decision boundaries.

---

# 4. Passive Aggressive Classifier

Passive Aggressive learning was used because:
- it performs efficiently on sparse textual data,
- supports large-scale online classification,
- updates weights aggressively only when misclassification occurs.

---

# Evaluation Metrics

The models were evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Since the dataset was imbalanced, emphasis was placed on:
- Recall,
- Macro F1-score,
- minority-class performance

rather than relying solely on accuracy.

---

# Final Experimental Results

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| MultinomialNB | 83.86% | 85.66% | 94.33% | 89.78% |
| LogisticRegression | 81.03% | 91.09% | 82.89% | 86.79% |
| PassiveAggressive | 79.98% | 87.45% | 85.67% | 86.55% |
| LinearSVC | 79.91% | 90.42% | 81.97% | 85.99% |

---

# Key Findings

## MultinomialNB Achieved Best Overall Performance

The Naive Bayes classifier demonstrated superior performance because:
- TF-IDF lexical statistics were highly informative,
- sparse probabilistic modeling aligned well with dataset characteristics.

---

## Logistic Regression Provided Strong Precision

Balanced Logistic Regression achieved:
- high precision,
- balanced classification,
- improved minority-class sensitivity.

---

## Importance of Imbalance Handling

Introducing class weighting significantly improved:
- fake-news recall,
- minority sensitivity,
- reduction of misinformation escape.

---

## Accuracy Alone Was Insufficient

The project demonstrated that:
- accuracy can be misleading in imbalanced datasets.

Hence:
- recall,
- F1-score,
- confusion matrix analysis

were essential for proper evaluation.

---

# Research Insights

The project revealed several important NLP and machine learning insights.

## Classical NLP Models Remain Powerful

Despite modern deep learning popularity, classical approaches such as:
- TF-IDF + Naive Bayes,
- TF-IDF + Logistic Regression

still perform strongly on structured textual classification tasks.

---

## Sparse Lexical Features are Highly Informative

The dataset contained strong lexical separability between fake and real news.

This enabled classical machine learning algorithms to achieve competitive performance.

---

## Limitations of Classical Models

Although effective, TF-IDF based models:
- rely on sparse lexical statistics,
- lack contextual semantic understanding,
- cannot effectively model long-range dependencies.

---

# Future Scope

Future improvements may include:
- Word2Vec embeddings
- GloVe embeddings
- LSTM networks
- Transformer architectures such as BERT and RoBERTa

Transformer models can better capture:
- semantic meaning,
- contextual dependencies,
- nuanced misinformation patterns,
- linguistic structure.

---

# Final Conclusion

This project successfully implemented a complete NLP-based fake news detection pipeline using classical machine learning techniques.

The work included:
- preprocessing,
- TF-IDF feature engineering,
- imbalance-aware learning,
- comparative classifier benchmarking,
- detailed performance evaluation.

Among all evaluated models, Multinomial Naive Bayes achieved the best overall performance on sparse TF-IDF representations, while Logistic Regression and LinearSVC demonstrated strong precision-oriented behavior.

The project highlights both:
- the effectiveness of classical NLP techniques,
- and the need for transformer-based contextual models for future advancement in misinformation detection systems.