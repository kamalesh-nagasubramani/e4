import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os

# --- Student Information ---
STUDENT_ROLL_NUMBER = "ENTER_ROLL_NUMBER_HERE"
print(f"Student Roll Number: {STUDENT_ROLL_NUMBER}")

# 2. Load the SMS Spam dataset from local archive (8)
dataset_path = r"C:\Users\kamal\Downloads\archive (8)\spam.csv"

# The dataset uses latin-1 encoding and has extra unnamed columns
try:
    df = pd.read_csv(dataset_path, encoding='latin-1')
    # Drop extra columns and rename
    df = df[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'message'})
except Exception as e:
    print(f"Error loading local dataset: {e}")
    # Fallback to download if local fails for some reason
    print("Attempting to download dataset as fallback...")
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
    if not os.path.exists('SMSSpamCollection'):
        import urllib.request
        import zipfile
        urllib.request.urlretrieve(url, 'sms.zip')
        with zipfile.ZipFile('sms.zip', 'r') as zip_ref:
            zip_ref.extractall('.')
        os.remove('sms.zip')
    df = pd.read_csv('SMSSpamCollection', sep='\t', names=['label', 'message'])

print("\nDataset Head:")
print(df.head())

# 3. Perform data preprocessing
try:
    nltk.download('stopwords')
except:
    pass
stop_words = set(stopwords.words('english'))

def clean_text(text):
    # Lowercase
    text = str(text).lower()
    # Remove punctuation
    text = "".join([char for char in text if char not in string.punctuation])
    # Remove stopwords
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

print("\nPreprocessing text...")
df['clean_message'] = df['message'].apply(clean_text)

# 4. Convert text into numerical features
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df['clean_message'])

# 5. Encode target labels
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
y = df['label_num']

# 6. Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Train a Multinomial Naive Bayes classifier
print("\nTraining Multinomial Naive Bayes...")
mnb = MultinomialNB()
mnb.fit(X_train, y_train)

# 8. Predict message classes
y_pred = mnb.predict(X_test)

# 9. Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\nAccuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# 10. Analyze misclassified examples
print("\nMisclassified Examples Analysis:")
test_indices = y_test.index
misclassified_mask = y_pred != y_test
misclassified_indices = test_indices[misclassified_mask]

# Get the predictions for these indices in order
y_test_arr = y_test.values
y_pred_arr = y_pred
mis_indices_loc = np.where(misclassified_mask)[0]

for i, loc in enumerate(mis_indices_loc[:5]):
    idx = test_indices[loc]
    print(f"Original: {df.iloc[idx]['message']}")
    print(f"Actual: {df.iloc[idx]['label']}, Predicted: {'spam' if y_pred_arr[loc] == 1 else 'ham'}")
    print("-" * 30)

# 11. Apply Laplace smoothing and observe impact
print("\nObserving impact of Laplace smoothing (alpha):")
alphas = [0.1, 0.5, 1.0, 2.0, 5.0]
for a in alphas:
    model = MultinomialNB(alpha=a)
    model.fit(X_train, y_train)
    p = model.predict(X_test)
    print(f"Alpha: {a}, Accuracy: {accuracy_score(y_test, p):.4f}")

# --- Visualizations ---
print("\nGenerating Visualizations...")

# Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - SMS Spam Classification')
plt.savefig('confusion_matrix_mnb.png')
plt.close()

# Feature importance (Top words influencing spam classification)
feature_names = tfidf.get_feature_names_out()
log_prob = mnb.feature_log_prob_[1] 
sorted_features = sorted(zip(log_prob, feature_names), reverse=True)
top_20_spam = sorted_features[:20]

prob, words = zip(*top_20_spam)
plt.figure(figsize=(10, 8))
sns.barplot(x=list(prob), y=list(words), palette='Reds_d')
plt.title('Top 20 Words Influencing Spam Classification (Log Prob)')
plt.savefig('feature_importance_mnb.png')
plt.close()

# Word frequency comparison (Spam vs Ham)
from collections import Counter
spam_words = " ".join(df[df['label'] == 'spam']['clean_message']).split()
ham_words = " ".join(df[df['label'] == 'ham']['clean_message']).split()

spam_counts = Counter(spam_words).most_common(20)
ham_counts = Counter(ham_words).most_common(20)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

sw, sc = zip(*spam_counts)
sns.barplot(x=list(sc), y=list(sw), ax=ax1, palette='Oranges_r')
ax1.set_title('Top 20 Words in Spam Messages')

hw, hc = zip(*ham_counts)
sns.barplot(x=list(hc), y=list(hw), ax=ax2, palette='Greens_r')
ax2.set_title('Top 20 Words in Ham Messages')

plt.tight_layout()
plt.savefig('word_frequency_comparison.png')
plt.close()

print("\nProcessing Complete. Visualizations saved as PNG files.")
