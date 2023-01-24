"""
@author: dsmueller3760
Tools to analyzed open ended answers.
"""
import numpy as np
import pandas as pd
import openai
from openai.embeddings_utils import cosine_similarity, get_embedding
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import seaborn as sns; sns.set()
import matplotlib
import matplotlib.pyplot as plt
import statistics

class OpenEndedAnswer:
    """ An open ended answer object. """
    def __init__(self, df, metrics):
        self.df = df
        self.metrics = metrics
        
        self.n_clusters = None
        self.cluster_descriptions = None
        self.matrix = None
        
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.rfr = None
        self.rfr = None
        
        self.n_clusters = None
        self.cluster_descriptions = None
        self.matrix = None
    def __str__(self):       
        """ Prints out key info about the OpenEndedAnswer object."""
        str_out = ''
        # Print Question_ID and question. Also number of answers provided to train.
        str_out += 'Question: '+str(self.df['Question_ID'].iloc[0])+', '+self.df['Question'].iloc[0]+'\n'
        str_out+= f'# of Answers in Model: {self.df.shape[0]}\n'
        
        # Print metrics and the number of clusters
        # TODO: Add a case statement to only do this and next block if specified.
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
            embedding_model = 'text-embedding-ada-002'
            self.df = self.df.assign(embedding = self.df['Answer'].apply(lambda x: get_embedding(x, engine=embedding_model)))
            self.df.to_csv(file[:-4]+'_with_embeddings.csv')
        else:
            # Create trained model with graded answers and embeddings
            # Read in the data with embeddings. This only works if you have run generate embeddings at least once.
            self.df = pd.read_csv(file[:-4]+'_with_embeddings.csv')
            self.df['embedding'] = self.df.embedding.apply(
                eval).apply(np.array)
        self.X_train = []
        self.X_test = []
        self.y_train = []
        self.y_test = []
        self.rfr = []
        # Loop through metrics and generate machine learning models for each of them per question.
        for metric in self.metrics:
            # Train model to predict categories
            X_train, X_test, y_train, y_test = train_test_split(list(
                self.df.embedding.values), 
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
    def generate_answer_embeddings(self, file, 
                            generate_embeddings=False, 
                            embedding_model='text-embedding-ada-002',
                            random_state=None,
                            debug=False):
        """ Generates answer embeddings. Saves them to a csv file if they are new or reads existing ones if specified. """
        if generate_embeddings:
            # Generates embeddings of the answers by rereading the original data and rewriting to new with_embeddings
            self.df = self.df.assign(embedding = self.df['Answer'].apply(lambda x: get_embedding(x, engine=embedding_model)))
            self.df.to_csv(file[:-4]+'_with_embeddings.csv')
        else:
            # Create trained model with graded answers and embeddings
            # Read in the data with embeddings. This only works if you have run generate embeddings at least once.
            self.df = pd.read_csv(file[:-4]+'_with_embeddings.csv')
            self.df['embedding'] = self.df.embedding.apply(
                eval).apply(np.array)
    def test_model(self,input_answer):
        """ Predict score of metrics for input_answer to the question """
        #  Create embedding from input answer
        input_embedding = get_embedding(input_answer, engine='text-embedding-ada-002')
        
        prediction = []
        for i in range(len(self.metrics)):
            prediction.append(self.rfr[i].predict([input_embedding]))
            # mse = mean_squared_error(self.y_test[i], preds)
            # mae = mean_absolute_error(self.y_test[i], preds)
        return prediction
    def make_named_clusters(self, n_clusters=3, 
                      random_state=None, 
                      ans_per_cluster=3, 
                      cluster_description_file=None,
                      debug=False):
        """ Creates named clusters based on a number of clusters requested """
        # Name the clusters with openai and show the similar answers
        self.n_clusters = n_clusters

        # Generate clusters
        matrix = np.vstack(self.df.embedding.values)
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
    def plot_pairs(self, fig_path = None):
        """ Create a pairplot to visualize the scores of answers """
        pair_plot = sns.pairplot(self.df, height=5, 
                                 vars = self.metrics,
                                 diag_kind = None,
                                 kind = 'hist',
                                 corner = True)
        
        if fig_path:
            pair_plot.savefig(fig_path)
    def plot_graded_clusters(self, fig_path = None, random_state=None):
        """ Plots embeddings colored by rating. Generates 1 x plot per metric. """
        
        # Commented out line was in ther example, but I think it doesn't work.
        # matrix = np.array(self.df.embedding.apply(eval).to_list())
        matrix = np.vstack(self.df.embedding.values)
        
        # Create a t-SNE model and transform the data
        tsne = TSNE(n_components=2, perplexity=15, random_state=random_state, init='random', learning_rate=200)
        vis_dims = tsne.fit_transform(matrix)
        colors = ['red', 'orange', 'green']
        x = [x for x,y in vis_dims]
        y = [y for x,y in vis_dims] 
        colormap = matplotlib.colors.ListedColormap(colors)
        
        # Create a new plot per metric, colored by the grade.
        # TODO: Add outlier detection in this plot: https://towardsdev.com/outlier-detection-using-k-means-clustering-in-python-214188fc90e8
        # TODO: Only works where grading is discrete -1, 0, 1. Make continuous.
        for metric in self.metrics:
            plt.figure()
            plt.rcParams['figure.figsize'] = [15.0, 15.0]
            plt.rcParams['figure.dpi'] = 140
            color_indices = getattr(self.df,metric)+1
            plt.scatter(x, y, c=color_indices, cmap=colormap, alpha=0.3)
            plt.title(f'{metric} visualized in language using t-SNE')
            for j in range(len(x)):
                plt.text(x=x[j]+0.3,y=y[j]+0.3,s=self.df.index[j])
            if fig_path:
                plt.savefig(fig_path[:-4]+f'_{metric}.png')
    def plot_named_clusters(self, fig_path = None, random_state = None):
        """ Clusters identified visualized in language 2d using t-SNE """
        # Requires that you first run make_clusters()
        tsne = TSNE(
            n_components=2, perplexity=15, random_state=random_state, init="random", learning_rate=200
        )
        vis_dims2 = tsne.fit_transform(self.matrix)

        x = [x for x, y in vis_dims2]
        y = [y for x, y in vis_dims2]
        i = 0

        # TODO: Modify this to plot arbitrary number. This only actually makes the same numer as the number of colors.
        plt.figure()
        plt.rcParams['figure.figsize'] = [15.0, 15.0]
        plt.rcParams['figure.dpi'] = 140
        for category, color in enumerate(['red', 'orange', 'green']):
            df_cluster = self.df[self.df.Cluster == category]
            xs = np.array(x)[self.df.Cluster == category]
            ys = np.array(y)[self.df.Cluster == category]
            # Plot datapoints. Plot mean data.
            plt.scatter(xs, ys, color=color, alpha=0.3)
            avg_x = xs.mean()
            avg_y = ys.mean()
            plt.scatter(avg_x, avg_y, marker='x', color=color,
                        s=100, label=f'Cluster {i}')
            plt.legend()
            # Apply data labels of the Question_ID to each datapoint.
            for j in range(len(xs)):
                plt.text(x=xs[j]+0.3,y=ys[j]+0.3,s=df_cluster.index[j])
            i = i+1
        if fig_path:
            plt.savefig(fig_path)
    def plot_cluster_efficiency():
        # TODO: Elbow plot or something like that
        return False
class OpenEndedMetric:
    """ An open ended metric object, which is intended to be used with zero shot approach """
    def __init__(self, df):
        self.metric = df['Metric'].iloc[0]
        self.df = df
    def __str__(self):            
        """ Prints out key info for OpenEndedMetric """
        str_out = ''
        # Print metrics dataframe
        str_out += 'Metrics: '+str(self.df)+'\n'
    def generate_metric_raw_embeddings(self, file,
                                   generate_embeddings=False, 
                                   embedding_model='text-embedding-ada-002'):
        """ Generates raw embeddings for exactly the metric as-written. Is not question specific """
        if generate_embeddings: 
            # Generates embeddings of the answers by rereading the original data and rewriting to new with_embeddings
            self.df = self.df.assign(embedding_pos = self.df['Category_term_pos'].apply(lambda x: get_embedding(x, engine=embedding_model)))
            self.df = self.df.assign(embedding_neg = self.df['Category_term_neg'].apply(lambda x: get_embedding(x, engine=embedding_model)))
            self.df.to_csv(file[:-4]+'_with_embeddings.csv')
            print(f'Embeddings created for {self.df.Metric.iloc[0]} category terms.')
        else:
            # TODO: no clue if this works
            # Create trained model with graded answers and embeddings
            # Read in the data with embeddings. This only works if you have run generate embeddings at least once.
            self.df = pd.read_csv(file[:-4]+'_with_embeddings.csv')
            self.df['embedding'] = self.df.embedding.apply(
                eval).apply(np.array)
            print(f'Embeddings file read for {self.df.Metric.iloc[0]} category terms.')
    def generate_metric_question_embeddings(self, ansObj, file,
                                   generate_embeddings=False, 
                                   embedding_model='text-embedding-ada-002',):
        """ Generates embeddings for the metric as-written in the context of the question to compare against.
            Is question specific """
        if generate_embeddings: 
            # Generates embeddings of the answers by rereading the original data and rewriting to new with_embeddings
            question = ansObj.df['Question'].iloc[0]
            category_term_pos = []
            category_term_neg = []
            embedding_pos = []
            embedding_neg = []
            for i in range(self.df.shape[0]):
                category_term_pos_txt = f'A {self.df.Category_term_short_pos.iloc[i]} answer to the following interview question:\n{question}\n'
                category_term_neg_txt = f'A {self.df.Category_term_short_neg.iloc[i]} answer to the following interview question:\n{question}\n'
                category_term_pos.append(category_term_pos_txt)
                category_term_neg.append(category_term_neg_txt)
                embedding_pos.append(get_embedding(category_term_pos_txt, engine=embedding_model))
                embedding_neg.append(get_embedding(category_term_neg_txt, engine=embedding_model))
            self.df = self.df.assign(Category_term_pos = category_term_pos)
            self.df = self.df.assign(Category_term_neg = category_term_neg)
            self.df['embedding_pos'] = embedding_pos
            self.df['embedding_neg'] = embedding_neg
            self.df.to_csv(file[:-4]+'_with_embeddings.csv')
            print(f'Question specific embeddings created for {self.df.Metric.iloc[0]} category terms.')
        else:
            # TODO: no clue if this works
            # Create trained model with graded answers and embeddings
            # Read in the data with embeddings. This only works if you have run generate embeddings at least once.
            self.df = pd.read_csv(file[:-4]+f'_qID_{ansObj.df.Question_ID.iloc[0]}_with_embeddings.csv')
            self.df['embedding'] = self.df.embedding.apply(
                eval).apply(np.array)
            print(f'Embeddings file read for {self.df.Metric.iloc[0]} category terms.')

def metric_score(metObj,ansObj):
    """ Calculates the cosine similarity between a metric and the answer
        More details here: https://community.openai.com/t/embeddings-and-cosine-similarity/17761/13 """
    data_out = []
    for i in range(ansObj.df.shape[0]):
        temp = [0]*metObj.df.shape[0]
        for j in range(metObj.df.shape[0]):
            temp[j] = cosine_similarity(ansObj.df['embedding'].iloc[i], metObj.df['embedding_pos'].iloc[j])-cosine_similarity(ansObj.df['embedding'].iloc[i], metObj.df['embedding_neg'].iloc[j])
        data_out.append(temp)
    data_out = data_out / np.amax(data_out)
    return data_out
def plot_embedding_metric_results(metObj, ansObj, score = None, fig_path = None):
    """ Creates plots of metric_score results, which are used to compare manual scoring """
    idx_sorted = np.argsort(getattr(ansObj.df,metObj.df['Metric'].iloc[0]))
    score_averages = []
    score_error = []
    if not score:
        score = metric_score(metObj,ansObj)
    score = score[idx_sorted]
    for ans_score in score:
        score_averages.append(statistics.mean(ans_score))
        score_error.append((max(ans_score)-min(ans_score))/2)

    score_diff = score_averages - getattr(ansObj.df,metObj.df['Metric'].iloc[0])

    # Sort the answers to show lowest to highest metric grade
    x_label_sort = list(map(str, ansObj.df.index.values.tolist()))
    ansObj.df = ansObj.df.iloc[idx_sorted]
    
    # Build the bar plot
    fig, ax = plt.subplots(figsize=(15,5))
    ax.bar(x_label_sort, score_averages, yerr=score_error, align='center', ecolor='black')
    ax.set_ylabel('Metric correlation')
    ax.set_xticks(x_label_sort)
    ax.set_xticklabels(x_label_sort)
    ax.set_title(f'Metric correlation: Question {ansObj.df.Question_ID.iloc[0]}, Metric {metObj.df.Metric.iloc[0]}')
    ax.yaxis.grid(True)
    # Overlay the anticipated scoring I did manually
    ax.scatter(x_label_sort, getattr(ansObj.df,metObj.df['Metric'].iloc[0]), color='k', label='Manual Grading')
    plt.savefig(fig_path+f'embedding_metric_correlation_qID{ansObj.df.Question_ID.iloc[0]}_{metObj.df.Metric.iloc[0]}.png')    
    plt.show()

    # Plot the difference between embeddings with zero shot and manual grading
    fig, ax = plt.subplots(figsize=(15,5))
    ax.bar(x_label_sort, score_diff, label='Manual Grading')
    ax.set_ylabel('Zero shot - manual correlation')
    ax.set_xticks(x_label_sort)
    ax.set_xticklabels(x_label_sort)
    ax.set_title(f'Zero shot - manual correlation: {ansObj.df.Question_ID.iloc[0]}, Metric {metObj.df.Metric.iloc[0]}')
    ax.yaxis.grid(True)
    # Overlay the anticipated scoring I did manually
    ax.scatter(x_label_sort, getattr(ansObj.df,metObj.df['Metric'].iloc[0]), color='k', label='Manual Grading')
    plt.savefig(fig_path+f'zero_shot_to_manual_qID{ansObj.df.Question_ID.iloc[0]}_{metObj.df.Metric.iloc[0]}.png')    
    plt.show()