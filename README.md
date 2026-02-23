SCENARIO 1 – MULTINOMIAL NAÏVE BAYES


Problem Statement

Classify SMS messages as Spam or Ham (Not Spam)..


Dataset (Kaggle – Public) https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset


Target Variable: Message Label (Spam / Ham)

Input Feature: SMS Text Messages


IN-LAB TASKS (Multinomial Naïve Bayes)

1. Import required Python libraries.

2. Load the SMS Spam dataset.

3. Perform data preprocessing:

· Text cleaning (lowercase, punctuation removal)

· Stopword removal (optional)

4. Convert text into numerical features using:

· Count Vectorization / TF-IDF

5. Encode target labels.

6. Split dataset into training and testing sets.

7. Train a Multinomial Naïve Bayes classifier.

8. Predict message classes.

9. Evaluate performance using:

· Accuracy

· Precision

· Recall

· F1 Score

10. Analyze misclassified examples.

11. Apply Laplace smoothing and observe impact.


Visualization

• Confusion Matrix

• Feature importance (Top words influencing spam classification)

• Word frequency comparison (Spam vs Ham)






SCENARIO 2 – GAUSSIAN NAÏVE BAYES


Problem Statement

Classify flower species based on physical measurements.


Dataset (Public / Standard Dataset)

Iris Dataset (sklearn)


Target Variable: Flower Species

Input Features

• Sepal Length • Sepal Width • Petal Length • Petal Width


IN-LAB TASKS (Gaussian Naïve Bayes)

1. Import required Python libraries.

2. Load the Iris dataset.

3. Perform data inspection & preprocessing.

4. Apply feature scaling.

5. Split dataset into training and testing sets.

6. Train a Gaussian Naïve Bayes classifier.

7. Predict species labels.

8. Evaluate performance using:

· Accuracy

· Precision / Recall / F1 Score

9. Compare predictions with actual labels.

10. Analyze class probabilities.

11. Compare Gaussian NB with Logistic Regression (optional).


Visualization

• Decision Boundary Plot (using two features)

• Confusion Matrix

• Probability distribution plots


SUBMISSION REQUIREMENTS

· Python code (with student roll numbers)

· Screenshot of code

· Screenshot of outputs and graphs

· GitHub repository link
