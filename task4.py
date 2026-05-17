import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
import re
df = pd.read_csv("sentimentdataset.csv")
print(df.head())
def clean_text(text):
    text = str(text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    return text.lower()

df['Cleaned_Text'] = df['Text'].apply(clean_text)
def get_sentiment(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity

    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

df['Sentiment'] = df['Cleaned_Text'].apply(get_sentiment)

print(df[['Text', 'Sentiment']].head())

sentiment_counts = df['Sentiment'].value_counts()

print("\nSentiment Counts:")
print(sentiment_counts)

plt.figure(figsize=(6,4))
sentiment_counts.plot(kind='bar')

plt.title("Sentiment Analysis Results")
plt.xlabel("Sentiment")
plt.ylabel("Count")

plt.show()
