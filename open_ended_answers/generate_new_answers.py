"""
Creates answer data from input questions.

https://openai.com/blog/introducing-text-and-code-embeddings/
Must have API key set to run.
"""

import pandas as pd
import open_ended_tools

#%% Define run parameters
directory = './Data/'
file_answers = 'open_ended_answers.csv'
embedding_model='text-embedding-ada-002'

#%% Read Data
df = pd.read_csv(directory+file_answers, index_col=0)
df = df[['Question_ID', 'Type', 'Question', 'Answer', 'Correct_answer',
         'Curiosity', 'Hunger', 'Smarts','Relevance',
         'Curiosity_optimum', 'Hunger_optimum', 'Smarts_optimum', 'Relevance_optimum']]

#%% Make answers
n_answers = 5
df_new_ans = open_ended_tools.make_answers(df,q_ID=None,n_answers=n_answers)
df_new_ans.to_csv(directory+file_answers[:-4]+'_new_answers.csv')