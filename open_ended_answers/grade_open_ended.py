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
file = 'open_ended_answers.csv'
generate_embeddings = False
n_clusters = 5  # Determined the number of clusters to use
# Set to none to truly randomize. 42 used in code to reproduce samples to what is in openai docs.
random_state = 10

# Read Data
df = pd.read_csv(directory+file, index_col=0)
df = df[['Question ID', 'Question', 'Time', 'Answer',
         'Curiousity_grade', 'Hunger_grade', 'Smarts_grade',
         'Curiousity_optimum', 'Hunger_optimum', 'Smarts_optimum']]
df = df.dropna()
metrics = ['Curiousity_grade', 'Hunger_grade', 'Smarts_grade']
# df.groupby('Question ID')

# Create open_ended_answer object and run functions
question = []
ans = []
for i in range(len(df['Question ID'].unique())):
    q_ID = df['Question ID'].unique()[i]
    question.append(df['Question'][df.index[df['Question ID'] == q_ID].tolist()[0]])
    ans.append(open_ended_tools.OpenEndedAnswer(df[df['Question ID'] == q_ID], metrics))
    
    ans[i].create_answer_model(directory+file[:-4]+f'_{q_ID}.csv', 
                               random_state=random_state, 
                               generate_embeddings=generate_embeddings)
    ans[i].make_clusters(n_clusters=n_clusters, 
                         random_state=random_state, 
                         ans_per_cluster=1,
                         cluster_description_file=directory+file[:-4]+f'_{q_ID}_cd.csv')
    print(ans[i])
    ans[i].plot_clusters(random_state=random_state, fig_path=directory+file[:-4]+f'_{q_ID}.png')