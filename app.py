import pandas as pd

file_path = 'dataset-sms-penipuan.csv'

data = pd.read_csv(file_path)

print(data.head())

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report

X_train, X_test, y_train, y_test = train_test_split(data['message'], data['type'], test_size=0.2, random_state=42)

vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

nb_classifier = MultinomialNB()

nb_classifier.fit(X_train_vec, y_train)

y_pred = nb_classifier.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

new_message = "Selamat kamu dapat kesempatan mendapatkan uang tunai sebesar 10juta rupiah!"

new_message_vec = vectorizer.transform([new_message])

predicted_label = nb_classifier.predict(new_message_vec)

print("Predicted Label:", predicted_label)