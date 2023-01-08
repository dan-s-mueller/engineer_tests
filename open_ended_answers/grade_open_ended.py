#%% Generate Embeddings
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 20:29:20 2023

@author: dsmueller3760
https://openai.com/blog/introducing-text-and-code-embeddings/
Must have API key set to run.
"""

from openai.embeddings_utils import get_embedding
import pandas as pd

# Read Data
input_datapath = './Data_Training/open_ended_answers.csv'
df = pd.read_csv(input_datapath, index_col=0)
df = df[['Question ID','Question','Time','Answer',
         'Curiousity_grade','Hunger_grade','Smarts_grade',
         'Curiousity_optimum','Hunger_optimum','Smarts_optimum']]
df = df.dropna()

directory="./Data_Training/"
generate_embeddings=False
random_state=42 #Set to none to truly randomize. 42 used in code to reproduce samples to what is in openai docs.
n_clusters = 4  # Determined the number of clusters to use

if generate_embeddings:
    # Generates embeddings of the answers by rereading the original data and rewriting to new with_embeddings
    df['ada_similarity'] = df.Answer.apply(lambda x: get_embedding(x, engine='text-embedding-ada-002'))
    df['ada_search'] = df.Answer.apply(lambda x: get_embedding(x, engine='text-embedding-ada-002'))
    df.to_csv('./Data_Training/open_ended_answers_with_embeddings.csv')
    print("Embeddings created.")

#%% Create trained model with graded answers and embeddings
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Read in the data with embeddings
df = pd.read_csv('./Data_Training/open_ended_answers_with_embeddings.csv')
df["ada_similarity"] = df.ada_similarity.apply(eval).apply(np.array)
metrics=['Curiousity_grade','Hunger_grade','Smarts_grade']

X_train_metrics=[]
X_test_metrics=[]
y_train_metrics=[]
y_test_metrics=[]
rfr_metrics=[]

for metric in metrics:
    # Train model to predict categories
    X_train, X_test, y_train, y_test = train_test_split(list(df.ada_similarity.values), getattr(df,metric), test_size=1, random_state=random_state)
    X_train_metrics.append(X_train)
    X_test_metrics.append(X_test)
    y_train_metrics.append(y_train)
    y_test_metrics.append(y_test)
    
    rfr = RandomForestRegressor(n_estimators=100)
    rfr_metrics.append(rfr)
    rfr.fit(X_train, y_train)
    preds = rfr.predict(X_test)
    
    mse = mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    
    print(f"Ada similarity embedding performance of {metric}: mse={mse:.2f}, mae={mae:.2f}")
    
    bmse = mean_squared_error(y_test, np.repeat(y_test.mean(), len(y_test)))
    bmae = mean_absolute_error(y_test, np.repeat(y_test.mean(), len(y_test)))
    print(
        f"Dummy mean prediction performance for {metric}: mse={bmse:.2f}, mae={bmae:.2f}\n"
    )
    
#%% Cluster the data
# Clusters are totally unrelated to the categories provided.
import numpy as np

matrix = np.vstack(df.ada_similarity.values)
matrix.shape

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=random_state, n_init=10)
kmeans.fit(matrix)
labels = kmeans.labels_
df["Cluster"] = labels

print(df.groupby("Cluster").Curiousity_grade.mean())
print(df.groupby("Cluster").Hunger_grade.mean())
print(df.groupby("Cluster").Smarts_grade.mean())

#%% Name the clusters with openai and show the similar answers
import openai

# Reading a review which belong to each group.
ans_per_cluster = 3

for i in range(n_clusters):
    print(f"Cluster {i} Theme:", end=" ")

    answers = "\n".join(
        df[df.Cluster == i]
        .Answer
        .sample(ans_per_cluster, random_state=42)
        .values
    )
    response = openai.Completion.create(
        engine="davinci-instruct-beta-v3",
        prompt=f'What do the following customer answers have in common?\n\nCustomer answers:\n"""\n{answers}\n"""\n\nTheme:',
        temperature=0,
        max_tokens=64,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    print(response["choices"][0]["text"].replace("\n", ""))

    sample_cluster_rows = df[df.Cluster == i].sample(ans_per_cluster, random_state=42)
    for j in range(ans_per_cluster):
        print(sample_cluster_rows.Curiousity_grade.values[j], end=", ")
        print(sample_cluster_rows.Answer.values[j], end=":   ")
        # print(sample_cluster_rows.Text.str[:70].values[j])

    print("-" * 100)
    
#%%Plot the data from clusters
def plot_clusters(matrix,fig_path):
    from sklearn.manifold import TSNE
    import matplotlib
    import matplotlib.pyplot as plt
    # Clusters identified visualized in language 2d using t-SNE
    tsne = TSNE(
        n_components=2, perplexity=15, random_state=random_state, init="random", learning_rate=200
    )
    vis_dims2 = tsne.fit_transform(matrix)
    
    x = [x for x, y in vis_dims2]
    y = [y for x, y in vis_dims2]
    i=0
        
    plt.figure(figsize=[12,12])
    for category, color in enumerate(["purple", "green", "red", "blue"]):
        xs = np.array(x)[df.Cluster == category]
        ys = np.array(y)[df.Cluster == category]
        plt.scatter(xs, ys, color=color, alpha=0.3)
    
        avg_x = xs.mean()
        avg_y = ys.mean()
    
        plt.scatter(avg_x, avg_y, marker="x", color=color, s=100,label=f"Cluster {i}")
        plt.legend()
        i=i+1
    plt.savefig(fig_path)
        
plot_clusters(matrix,"./Data_Training/question1_cluster.png")