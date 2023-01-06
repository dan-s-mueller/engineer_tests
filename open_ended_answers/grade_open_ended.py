# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 20:29:20 2023

@author: dsmueller3760
openai.api_key='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'
https://openai.com/blog/introducing-text-and-code-embeddings/
"""

from openai.embeddings_utils import get_embedding
import pandas as pd

# Read Data
input_datapath = './Data_Training/open_ended_answers.csv'
df = pd.read_csv(input_datapath, index_col=0)
df = df[['Question ID','Question','Time','Answer',
         'Curiousity, grade','Hunger, grade','Smarts, grade',
         'Curiousity, optimum','Hunger, optimum','Smarts, optimum']]
df = df.dropna()
df = df.sort_values('Time')


# Ensure you have your API key set in your environment per the README: https://github.com/openai/openai-python#usage
# This will take just between 5 and 10 minutes for 1k samples in the default example.
df['ada_similarity'] = df.Answer.apply(lambda x: get_embedding(x, engine='text-embedding-ada-002'))
df['ada_search'] = df.Answer.apply(lambda x: get_embedding(x, engine='text-embedding-ada-002'))
df.to_csv('./Data_Training/open_ended_answers_with_embeddings.csv')