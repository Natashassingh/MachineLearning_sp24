import pandas as pd
import nltk
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from tqdm import tqdm


# initial objects
nltk.download('wordnet')
nltk.download('stopwords')

lemmatizer = WordNetLemmatizer()
stopwords = set(stopwords.words('english'))
vectorizer = TfidfVectorizer(max_features=1000)  # lower max_features if you encounter memory issues


# building a processing function for the text to process product descriptions
def preprocess_desc(description):
    description = description.lower()
    description = re.sub(r'[^a-z0-9/]', ' ', description)
    token_list = description.split()
    lemmatized_list = [lemmatizer.lemmatize(word) for word in token_list if word not in stopwords]
    return ' '.join(lemmatized_list)


# read data and lemmatize data
print("Processing data")
tqdm.pandas()  # progress bar function
df = pd.read_csv('product_descriptions.csv')
df['lemmatized_desc'] = df['product_description'].progress_apply(preprocess_desc)
print("Data cleaned")

# build tf-idf matrix
tfidf_matrix = vectorizer.fit_transform(df['lemmatized_desc'])
print("TF-IDF matrix built")


# Cluster the similar products
num_clusters = 100
kmmModel = KMeans(n_clusters=num_clusters)
clusters = kmmModel.fit_predict(tfidf_matrix)
df['cluster_ID'] = clusters
print("Clusters built")

# send to a CSV
df.to_csv('cluster_data.csv', index=False)
print("Csv created.  Closing...")


# Display a sample of each cluster for debugging
for cluster_id in sorted(df['cluster_ID'].unique()):
    print(f"From cluster {cluster_id}:")
    cluster_snips = df[df['cluster_ID'] == cluster_id].head()

    for _, row in cluster_snips.iterrows():
        first_words = ' '.join(row['lemmatized_desc'].split()[:10])
        print(first_words)
    print("\n")

