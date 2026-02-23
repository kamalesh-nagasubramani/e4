import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# --- Student Information ---
STUDENT_ROLL_NUMBER = "ENTER_ROLL_NUMBER_HERE"
print(f"Student Roll Number: {STUDENT_ROLL_NUMBER}")

# 1. Import libraries (already done)

# 2. Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# 3. Perform data inspection & preprocessing
df = pd.DataFrame(X, columns=feature_names)
df['species'] = y
print("\nIris Dataset Inspection (Head):")
print(df.head())
print("\nDescription:")
print(df.describe())

# 4. Apply feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 6. Train a Gaussian Naive Bayes classifier
print("\nTraining Gaussian Naive Bayes...")
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# 7. Predict species labels
y_pred = gnb.predict(X_test)

# 8. Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
# For multi-class, we use 'weighted' or 'macro' averaging
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"\nGaussian NB Performance:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision (Weighted): {precision:.4f}")
print(f"Recall (Weighted): {recall:.4f}")
print(f"F1 Score (Weighted): {f1:.4f}")

# 9. Compare predictions with actual labels
comparison = pd.DataFrame({'Actual': [target_names[i] for i in y_test], 
                           'Predicted': [target_names[i] for i in y_pred]})
print("\nSample Comparisons (First 10):")
print(comparison.head(10))

# 10. Analyze class probabilities
probs = gnb.predict_proba(X_test)
print("\nClass Probabilities for First 5 Test Samples:")
prob_df = pd.DataFrame(probs, columns=target_names)
print(prob_df.head())

# 11. Compare Gaussian NB with Logistic Regression
print("\nTraining Logistic Regression for comparison...")
lr = LogisticRegression(random_state=42)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")

# --- Visualizations ---
print("\nGenerating Visualizations...")

# Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Iris Classification (Gaussian NB)')
plt.savefig('confusion_matrix_gnb.png')
plt.close()

# Decision Boundary Plot (using first two features: Sepal Length vs Sepal Width)
# We need to retrain on unscaled or scaled version of just those two features for 2D plotting
X_2d = X_scaled[:, :2] # Using first two scaled features
gnb_2d = GaussianNB()
gnb_2d.fit(X_2d, y)

x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

Z = gnb_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, edgecolors='k', cmap='viridis')
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
plt.title('Decision Boundary - Gaussian NB (Sepal features)')
plt.legend(handles=scatter.legend_elements()[0], labels=list(target_names))
plt.savefig('decision_boundary_gnb.png')
plt.close()

# Probability distribution plots (Violin plot of Petal Length for each class)
plt.figure(figsize=(10, 6))
sns.violinplot(x='species', y='petal length (cm)', data=df, inner="quartile")
plt.xticks(ticks=[0, 1, 2], labels=target_names)
plt.title('Distribution of Petal Length across Species')
plt.savefig('probability_distribution_gnb.png')
plt.close()

print("\nProcessing Complete. Visualizations saved as PNG files.")
