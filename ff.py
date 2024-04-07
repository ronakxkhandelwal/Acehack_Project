import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report

true_df = pd.read_csv('C:/Users/HP/Desktop/Code/Fake News Detection/True.csv')
true_df['label'] = 0  

fake_df = pd.read_csv('C:/Users/HP/Desktop/Code/Fake News Detection/Fake.csv')
fake_df['label'] = 1  

df = pd.concat([true_df, fake_df], ignore_index=True)

X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

model = make_pipeline(TfidfVectorizer(), MultinomialNB())

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
classification_report_result = classification_report(y_test, predictions)

print(f'Accuracy: {accuracy}')
print('Classification Report:\n', classification_report_result)

user_input = input("Enter a news article: ")

prediction = model.predict([user_input])

if prediction[0] == 0:
    print("The news is likely to be true.")
else:
    print("The news is likely to be fake.")
