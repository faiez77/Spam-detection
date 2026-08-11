import numpy as np
import pandas as pd
df = pd.read_csv("spam.csv", encoding="latin-1")
df = df[["v1", "v2"]]
df.columns = ["label", "text"]
df["label"] = df["label"].map({"ham": 0, "spam": 1})
df.head()
  
  // check for missing value
df.isnull().sum()   
  
 # class distribution
df["label"].value_counts
# convert to percentage
df["label"].value_counts(normalize=True)*100

# visualizing class imbalance
import matplotlib.pyplot as plt
df['label'].value_counts().plot(kind="bar")
plt.title("spam vs ham distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()

# spam messsage are usually longer
df["text_length"] = df["text"].apply(len)
df.groupby("label")["text_length"].mean()

# TEXT PREPROCESSING
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess(text):
    # Convert to lowercase
    text = text.lower()
    
    # Tokenize text
    tokens = word_tokenize(text)
    
    # Remove punctuation
    tokens = [word for word in tokens if word not in string.punctuation]
    
    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    
    # Join tokens back to sentence
    return " ".join(tokens)
df['clean_text']=df['text'].apply(preprocess)
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=5000)
# limit to top 5000word
X = tfidf.fit_transform(df['clean_text']).toarray()
y=df['label'].values

// model
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,test_size=0.2,          
    random_state=42,
     stratify=y              # keep spam/ham ratio same
)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(y_pred)
y_pred_prob=model.predict_proba(X_test)[:,1]
print(y_pred_prob)

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

print("Classification Report:",classification_report(y_test, y_pred))

print("Confusion Matrix:",confusion_matrix(y_test, y_pred))

print("ROC AUC Score:" , roc_auc_score(y_test , y_pred_prob))

sample_msg = ["Congratulations! You won a free gift card. Click now"]

# Preprocess
sample_clean = [preprocess(sample_msg[0])]

# TF-IDF vectorization
sample_vec = tfidf.transform(sample_clean)

# Predict probability
spam_prob = model.predict_proba(sample_vec)[0][1]

print("spam probablity:",spam_prob)

  # Optimizing thrshold (not using defaault 0.5 )
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

# Get predicted probabilities
y_scores = model.predict_proba(X_test)[:, 1]

# Compute precision, recall, thresholds
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)

# Plot
plt.figure(figsize=(8,5))
plt.plot(thresholds, precision[:-1], label='Precision')
plt.plot(thresholds, recall[:-1], label='Recall')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Precision-Recall vs Threshold')
plt.legend()
plt.show()

  optimal_threshold = 0.3  # if we want to catch more spam
y_pred_new = (y_scores >= optimal_threshold).astype(int)

# spam detection prefer recall over precision

  from sklearn.metrics import classification_report, confusion_matrix

print("Classification Report at threshold 0.3:")
print(classification_report(y_test, y_pred_new))
print("confusion Matrix",confusion_matrix(y_test , y_pred_new))

  from sklearn.metrics import roc_curve, roc_auc_score

fpr, tpr, roc_thresholds = roc_curve(y_test, y_scores)
auc = roc_auc_score(y_test, y_scores)

plt.figure(figsize=(7,5))
plt.plot(fpr, tpr, label=f'ROC curve (AUC={auc:.2f})')
plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
