#%% Generate Embeddings
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 20:29:20 2023

@author: dsmueller3760
https://openai.com/blog/introducing-text-and-code-embeddings/
Must have API key set to run.
"""
import numpy as np
import pandas as pd
import openai
from openai.embeddings_utils import get_embedding
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

class OpenEndedAnswer:
    def __init__(self, df, metrics):
        self.df = df
        self.metrics = metrics
                
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.ytest = None
        self.rfr = None
        self.rfr = None
        
        self.n_clusters = None
        self.cluster_descriptions = None
        self.matrix = None

    def __str__(self):            
        str_out = ''
        # Print question ID and question. Also number of answers provided to train.
        str_out += 'Question: '+str(self.df['Question ID'].iloc[0])+', '+self.df['Question'].iloc[0]+'\n'
        str_out+= f'# of Answers in Model: {self.df.shape[0]}\n'
        
        # Print metrics and the number of clusters
        str_out += 'Metrics: '+str(self.metrics)+'\n'
        str_out += f'# of Clusters: {self.n_clusters}\n'
        
        
        # Print the predictability of the model.
        for i in range(len(self.metrics)):
            preds = self.rfr[i].predict(self.X_test[i])
            mse = mean_squared_error(self.y_test[i], preds)
            mae = mean_absolute_error(self.y_test[i], preds)
            str_out += f'Ada similarity embedding performance of {self.metrics[i]}: mse={mse:.2f}, mae={mae:.2f}\n'
        return str_out

    def create_answer_model(self, file, 
                            generate_embeddings=False, 
                            random_state=None,
                            debug=False):
        # Creates an answer model by either generating embeddings, or using existing ones.
        # Outputs a RandomForestRegressor object which can be used to predict categories by metric.

        if generate_embeddings:
            # Generates embeddings of the answers by rereading the original data and rewriting to new with_embeddings
            self.df = self.df.assign(ada_similarity = self.df.Answer.apply(lambda x:
                                                             get_embedding(x, engine='text-embedding-ada-002')))
            self.df = self.df.assign(ada_search = self.df.Answer.apply(lambda x:
                                                         get_embedding(x, engine='text-embedding-ada-002')))
            self.df.to_csv(file[:-4]+'_with_embeddings.csv')
            print('Embeddings created.')
        else:
            # Create trained model with graded answers and embeddings
            # Read in the data with embeddings. This only works if you have run generate embeddings at least once.
            self.df = pd.read_csv(file[:-4]+'_with_embeddings.csv')
            self.df['ada_similarity'] = self.df.ada_similarity.apply(
                eval).apply(np.array)
            print('Embeddings file read.')

        
        self.X_train = []
        self.X_test = []
        self.y_train = []
        self.y_test = []
        self.rfr = []
        # Loop through metrics and generate machine learning models for each of them per question.
        for metric in self.metrics:
            # Train model to predict categories
            X_train, X_test, y_train, y_test = train_test_split(list(
                self.df.ada_similarity.values), 
                getattr(self.df, metric), test_size=1, random_state=random_state)
            
            self.X_train.append(X_train)
            self.X_test.append(X_test)
            self.y_train.append(y_train)
            self.y_test.append(y_test)

            rfr_temp = RandomForestRegressor(n_estimators=100,
                                             random_state=random_state)
            rfr_temp.fit(X_train, y_train)
            preds = rfr_temp.predict(X_test)
            self.rfr.append(rfr_temp)

            # Display some basic information about the predictability of the model for the metric.
            if debug:
                mse = mean_squared_error(y_test, preds)
                mae = mean_absolute_error(y_test, preds)
                print(
                    f"Ada similarity embedding performance of {metric}: mse={mse:.2f}, mae={mae:.2f}")
                
                bmse = mean_squared_error(
                    y_test, np.repeat(y_test.mean(), len(y_test)))
                bmae = mean_absolute_error(
                    y_test, np.repeat(y_test.mean(), len(y_test)))
                print(
                    f"Dummy mean prediction performance for {metric}: mse={bmse:.2f}, mae={bmae:.2f}\n"
                )

    def make_clusters(self, n_clusters=4, 
                      random_state=None, 
                      ans_per_cluster=3, 
                      cluster_description_file=None,
                      debug=False):
        # Name the clusters with openai and show the similar answers
        self.n_clusters = n_clusters

        # Generate clusters
        matrix = np.vstack(self.df.ada_similarity.values)
        matrix.shape
        self.matrix = matrix
        self.n_clusters = n_clusters
        kmeans = KMeans(n_clusters=n_clusters, init="k-means++",
                        random_state=random_state, n_init=10)
        kmeans.fit(matrix)
        labels = kmeans.labels_
        self.df = self.df.assign(Cluster = labels)
        
        if debug:
            for metric in self.metrics:
                print(getattr(self.df.groupby('Cluster'), metric).mean())

        responses = []
        example_answers = []
        # Reading a review which belong to each group.
        for i in range(n_clusters):
            if debug:
                print(f"Cluster {i} Theme:", end=" ")
            answers = "\n".join(
                self.df[self.df.Cluster == i]
                .Answer
                .sample(ans_per_cluster, random_state=random_state)
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
            responses.append(
                response["choices"][0]["text"].replace("\n", ""))
            if debug:
                print(responses[i])

            out_string = ""
            for j in range(ans_per_cluster):
                for metric in self.metrics:
                    metric_out = getattr(self.df[self.df.Cluster == i].sample(
                        ans_per_cluster, random_state=random_state), metric).values[j]
                    if debug:
                        print(metric_out, end=', ')
                    out_string = out_string + str(metric_out) + ', '
                ans_out = self.df[self.df.Cluster == i].sample(
                    ans_per_cluster, random_state=random_state).Answer.values[j]
                if debug:
                    print(ans_out, end='\n')
                out_string = out_string + ans_out + '\n'
            example_answers.append(out_string)
        self.cluster_descriptions = pd.DataFrame({'description': responses,
                                                  'example_answers': example_answers})
        if cluster_description_file:
            self.cluster_descriptions.to_csv(cluster_description_file)
    def plot_clusters(self, fig_path = None, random_state = None):
        # Clusters identified visualized in language 2d using t-SNE
        # Requires that you first run make_clusters()
        tsne = TSNE(
            n_components=2, perplexity=15, random_state=random_state, init="random", learning_rate=200
        )
        vis_dims2 = tsne.fit_transform(self.matrix)

        x = [x for x, y in vis_dims2]
        y = [y for x, y in vis_dims2]
        i = 0

        plt.figure(figsize=[18, 18])
        # TODO: Modify this to plot arbitrary number
        for category, color in enumerate(['purple', 'green', 'red', 'blue','black']):
            df_cluster = self.df[self.df.Cluster == category]
            xs = np.array(x)[self.df.Cluster == category]
            ys = np.array(y)[self.df.Cluster == category]
            plt.scatter(xs, ys, color=color, alpha=0.3)

            avg_x = xs.mean()
            avg_y = ys.mean()

            plt.scatter(avg_x, avg_y, marker='x', color=color,
                        s=100, label=f'Cluster {i}')
            plt.legend()
            
            for j in range(len(xs)):
                plt.text(x=xs[i]+0.3,y=ys[i]+0.3,s=df_cluster['Question ID'].iloc[j])
            i = i+1
        if fig_path:
            plt.savefig(fig_path)
    def test_model(self,input_answer):
        # Predict score of metrics for input_answer to the question
        
        # Create embedding from input answer
        input_embedding = get_embedding(input_answer, engine='text-embedding-ada-002')
        
        prediction = []
        for i in range(len(self.metrics)):
            prediction.append(self.rfr[i].predict([input_embedding]))
            # mse = mean_squared_error(self.y_test[i], preds)
            # mae = mean_absolute_error(self.y_test[i], preds)
        # return prediction
        return prediction