#%% Generate Embeddings
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 20:29:20 2023

@author: dsmueller3760
https://openai.com/blog/introducing-text-and-code-embeddings/
Must have API key set to run.
"""
import pandas as pd
import open_ended_tools

# Define run parameters
directory = './Data_Training/'
file_answers = 'open_ended_answers.csv'
file_metrics = 'metrics.csv'
generate_embeddings = False
embedding_model='text-embedding-ada-002'
n_clusters = 3  # Determined the number of clusters to use
# Set to none to truly randomize. 42 used in code to reproduce samples to what is in openai docs.
random_state = 40

# Read Data
df = pd.read_csv(directory+file_answers, index_col=0)
df = df[['Question_ID', 'Type', 'Question', 'Answer', 'Correct_answer',
         'Curiosity', 'Hunger', 'Smarts','Relevance',
         'Curiosity_optimum', 'Hunger_optimum', 'Smarts_optimum','Relevance_optimum']]

# metrics = ['Curiousity', 'Hunger', 'Smarts']
df_metrics = pd.read_csv(directory+file_metrics, index_col=0)
df_metrics = df_metrics[['Metric','Category_term_pos','Category_term_neg']]
metrics = df_metrics['Metric'].unique()


#%% Create open_ended_answer object and embeddings
question = []
ans = []
for i in range(len(df['Question_ID'].unique())):
    q_ID = df['Question_ID'].unique()[i]
    question.append(df['Question'][df.index[df['Question_ID'] == q_ID].tolist()[0]])
    ans.append(open_ended_tools.OpenEndedAnswer(df[df['Question_ID'] == q_ID], metrics))

    ans[i].generate_answer_embeddings(directory+file_answers[:-4]+f'_{q_ID}.csv', 
                                      random_state=random_state, 
                                      generate_embeddings=generate_embeddings,
                                      embedding_model=embedding_model)
    print(ans[i])
    
#%% Create metric embeddings
met = []
for i in range(len(df_metrics['Metric'].unique())):
    met_name = df_metrics['Metric'].unique()[i]
    met.append(open_ended_tools.OpenEndedMetric(df_metrics[df_metrics['Metric'] == met_name]))

    met[i].generate_metric_embeddings(directory+file_metrics,
                                      generate_embeddings=True, 
                                      embedding_model=embedding_model)

#%% Test out embeddings scoring
score = open_ended_tools.metric_score(met[0],ans[0])