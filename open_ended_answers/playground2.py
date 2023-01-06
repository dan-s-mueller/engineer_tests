# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 23:41:57 2022

@author: dsmue
openai.api_key='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'
https://beta.openai.com/docs/guides/embeddings/use-cases
https://github.com/openai/openai-cookbook/blob/main/examples/Obtain_dataset.ipynb
"""

# Start from obtain dataset notebook
from openai.embeddings_utils import get_embedding
import pandas as pd
 
# Read Data
input_datapath = './Data_Training/fine_food_reviews_1k.csv'  # to save space, we provide a pre-filtered dataset
df = pd.read_csv(input_datapath, index_col=0)
df = df[['Time', 'ProductId', 'UserId', 'Score', 'Summary', 'Text']]
df = df.dropna()
df['combined'] = "Title: " + df.Summary.str.strip() + "; Content: " + df.Text.str.strip()
df.head(2)

# subsample to 1k most recent reviews and remove samples that are too long
n_inputs=100
df = df.sort_values('Time').tail(n_inputs)
df.drop('Time', axis=1, inplace=True)

from transformers import GPT2TokenizerFast
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# remove reviews that are too long
df['n_tokens'] = df.combined.apply(lambda x: len(tokenizer.encode(x)))
df = df[df.n_tokens<8000].tail(n_inputs)
print(len(df))


# Ensure you have your API key set in your environment per the README: https://github.com/openai/openai-python#usage
# This will take just between 5 and 10 minutes for 1k samples in the default example.
df['ada_similarity'] = df.combined.apply(lambda x: get_embedding(x, engine='text-embedding-ada-002'))
df['ada_search'] = df.combined.apply(lambda x: get_embedding(x, engine='text-embedding-ada-002'))
df.to_csv('../openai-cookbook/examples/data/fine_food_reviews_with_embeddings_1k.csv')

# start of Regression using embeddings
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# If you have not run the "Obtain_dataset.ipynb" notebook, you can download the datafile from here: https://cdn.openai.com/API/examples/data/fine_food_reviews_with_embeddings_1k.csv
datafile_path = "../openai-cookbook/examples/data/fine_food_reviews_with_embeddings_1k.csv"

df = pd.read_csv(datafile_path)
df["ada_similarity"] = df.ada_similarity.apply(eval).apply(np.array)

X_train, X_test, y_train, y_test = train_test_split(list(df.ada_similarity.values), df.Score, test_size=1, random_state=42)

rfr = RandomForestRegressor(n_estimators=100)
rfr.fit(X_train, y_train)
preds = rfr.predict(X_test)

mse = mean_squared_error(y_test, preds)
mae = mean_absolute_error(y_test, preds)

print(f"Ada similarity embedding performance on 1k Amazon reviews: mse={mse:.2f}, mae={mae:.2f}")

bmse = mean_squared_error(y_test, np.repeat(y_test.mean(), len(y_test)))
bmae = mean_absolute_error(y_test, np.repeat(y_test.mean(), len(y_test)))
print(
    f"Dummy mean prediction performance on Amazon reviews: mse={bmse:.2f}, mae={bmae:.2f}"
)