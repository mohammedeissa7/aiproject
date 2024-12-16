# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset

data = pd.read_csv('StudentsPerformance.csv')

# Encode categorical variables
label_encoder = LabelEncoder()
for col in ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']:
    data[col] = label_encoder.fit_transform(data[col])

data['total_score'] = data['math score'] + data['reading score'] + data['writing score']
data['average_score'] = data['total_score'] / 3

'''
plt.figure(figsize=(12, 6))
sns.histplot(data['math score'], bins=30, kde=True, label='Math Score', color='blue')
sns.histplot(data['reading score'], bins=30, kde=True, label='Reading Score', color='green')
sns.histplot(data['writing score'], bins=30, kde=True, label='Writing Score', color='red')
plt.title("Distribution of Scores")
plt.legend()
plt.show()


plt.figure(figsize=(8, 5))
sns.countplot(x='gender', data=data, palette='coolwarm')
plt.title("Count of Students by Gender")
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x='gender', y='total_score', data=data, palette='viridis')
plt.title("Box Plot of Total Scores by Gender")
plt.show()


plt.figure(figsize=(10, 8))
correlation = data.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()

sns.pairplot(data[['math score', 'reading score', 'writing score', 'total_score']])
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot(x='test preparation course', data=data, palette='Set2')
plt.title("Count of Students by Test Preparation Course")
plt.show()

'''
# Target Variable: Classify students as High or Low performers
data['performance'] = np.where(data['average_score'] >= 60,1,0)  # 1 = High performer, 0 = Low performer

# Drop original scores
data = data.drop(['math score', 'reading score', 'writing score'], axis=1)

# Split features and target
X = data.drop('performance', axis=1)
y = data['performance']

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_resampled)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_resampled, test_size=0.4, random_state=42)


# Model Evaluation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
knn_model = KNeighborsClassifier(n_neighbors=21)
knn_model.fit(X_train, y_train)
y_pred = knn_model.predict(X_test)
print(classification_report(y_test, y_pred))

from sklearn.naive_bayes import GaussianNB
nb_model = GaussianNB()
nb_model = nb_model.fit(X_train, y_train)
y_pred = nb_model.predict(X_test)
print(classification_report(y_test, y_pred))

from sklearn.svm import SVC
svm_model = SVC()
svm_model = svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)
print(classification_report(y_test, y_pred))

from sklearn.linear_model import LogisticRegression
lg_model = LogisticRegression()
lg_model = lg_model.fit(X_train, y_train)
y_pred = lg_model.predict(X_test)
print(classification_report(y_test, y_pred))


