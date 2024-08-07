import sys
import pathlib
import re

import polars as pl
import matplotlib.pyplot as plt
from datasets import load_dataset
from wordcloud import WordCloud
from collections import Counter

# Check if a dataset location is provided as a command-line argument
if len(sys.argv) < 2:
    print("Usage: python GenerateWordCloud.py <Dataset Huggingface ID>")
    sys.exit(1)

dataset_name = sys.argv[1]

# Load the dataset from Hugging Face
dataset = load_dataset(dataset_name)

# Convert the dataset to a Polars DataFrame
data = pl.DataFrame(dataset['train'].to_pandas())

text_data =[]

for conversation in data['conversations'].to_list():
    if conversation is not None:
        for message in conversation:
            if message['from'] != 'system':
                text_data.append(message['value'])

# Combine all text into a single string
text = ' '.join(text_data)

# Clean the text (remove punctuation, lowercase)
text = re.sub(r'[^\w\s]', '', text.lower())

exclude_words = set(['i', 'you', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])

# Count word frequencies
word_freq = Counter(word for word in text.split() if word not in exclude_words)

# Create and generate a word cloud image
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)

# Display the generated image
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title(dataset_name)
plt.show()

script_dir = pathlib.Path(__file__).parent.absolute()
wordcloud_dir = script_dir / "wordclouds"

# Create the wordclouds directory if it doesn't exist
wordcloud_dir.mkdir(parents=True, exist_ok=True)

sanitized_dataset_name = dataset_name.replace('/', '_')

# Optionally, save the word cloud image
wordcloud.to_file(f"{wordcloud_dir}/{sanitized_dataset_name}-wordcloud.png")