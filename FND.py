import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the datasets
fake = pd.read_csv('Fake.csv')
real = pd.read_csv('True.csv')

# Add labels: 0 = Fake, 1 = Real
fake['label'] = 0
real['label'] = 1

# Combine and shuffle
df = pd.concat([fake[['text', 'label']], real[['text', 'label']]])
df = df.sample(frac=1).reset_index(drop=True)

X = df['text']
y = df['label']

# Convert text to numerical vectors
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_vec = vectorizer.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Accuracy and report
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Try your own text inputs
def test_news_input(news_text):
    input_vec = vectorizer.transform([news_text])
    prediction = model.predict(input_vec)
    return "REAL News" if prediction[0] == 1 else "FAKE News"

# Example usage:
print(test_news_input("Government announces free health checkups for all."))
print(test_news_input("Aliens spotted on Earth according to unnamed sources."))

