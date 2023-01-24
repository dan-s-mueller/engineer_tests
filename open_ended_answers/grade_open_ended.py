#%% Define inputs
"""
Created on Fri Jan  6 20:29:20 2023

@author: dsmueller3760
https://openai.com/blog/introducing-text-and-code-embeddings/
Must have API key set to run.
"""
import pandas as pd
import open_ended_tools

# Define run parameters
directory = './Data/'
file_answers = 'open_ended_answers.csv'
file_metrics = 'metrics_question_specific.csv'
generate_embeddings = True
embedding_model='text-embedding-ada-002'
n_clusters = 3  # Determined the number of clusters to use
random_state = 40   # Set to none to truly randomize. 42 used in code to reproduce samples to what is in openai docs.

# Read Data
df = pd.read_csv(directory+file_answers, index_col=0)
df = df[['Question_ID', 'Type', 'Question', 'Answer', 'Correct_answer',
         'Curiosity', 'Hunger', 'Smarts','Relevance',
         'Curiosity_optimum', 'Hunger_optimum', 'Smarts_optimum', 'Relevance_optimum']]

# metrics = ['Curiousity', 'Hunger', 'Smarts']
df_metrics = pd.read_csv(directory+file_metrics, index_col=0)
df_metrics = df_metrics[['Metric','Category_term_short_pos','Category_term_pos',
                         'Category_term_short_neg','Category_term_neg']]
metrics = df_metrics['Metric'].unique()


#%% Create open_ended_answer object and embeddings
question = []
ans = []
for i in range(len(df['Question_ID'].unique())):
    q_ID = df['Question_ID'].unique()[i]
    question.append(df['Question'][df.index[df['Question_ID'] == q_ID].tolist()[0]])
    ans.append(open_ended_tools.OpenEndedAnswer(df[df['Question_ID'] == q_ID], metrics))

    ans[i].create_answer_model(directory+file_answers[:-4]+f'_{q_ID}.csv', 
                                random_state=random_state, 
                                generate_embeddings=generate_embeddings,
                                embedding_model=embedding_model)
    ans[i].generate_answer_embeddings(directory+file_answers[:-4]+f'_{q_ID}.csv', 
                                      random_state=random_state, 
                                      generate_embeddings=generate_embeddings,
                                      embedding_model=embedding_model)
    print(ans[i])
    
#%%Create clusters, plot and analyze them
for i in range(len(df['Question_ID'].unique())):    
    q_ID = df['Question_ID'].unique()[i]
    ans[i].plot_pairs(fig_path=directory+file_answers[:-4]+f'_pp_{q_ID}.png')
    ans[i].make_named_clusters(n_clusters=n_clusters, 
                               random_state=random_state, 
                               ans_per_cluster=1,
                               cluster_description_file=directory+file_answers[:-4]+f'_{q_ID}_cd.csv')
    ans[i].plot_graded_clusters(random_state=random_state,
                                fig_path=directory+file_answers[:-4]+f'_graded_{q_ID}.png')
    ans[i].plot_named_clusters(random_state=random_state, fig_path=directory+file_answers[:-4]+f'_{q_ID}.png')
    
#%% Create question specific metric embeddings, score them and compre to manual grading.
#TODO: only works now for one question, this will use the first question in the answer set.
run = False
if run:
    met_qs = []
    for i in range(len(df_metrics['Metric'].unique())):
        met_name = df_metrics['Metric'].unique()[i]
        met_qs.append(open_ended_tools.OpenEndedMetric(df_metrics[df_metrics['Metric'] == met_name]))
    
        
        met_qs[i].generate_metric_question_embeddings(ans[0], directory+file_metrics,
                                          generate_embeddings=True, 
                                          embedding_model=embedding_model)
    for m in met_qs:
        open_ended_tools.plot_embedding_metric_results(m, ans[0], score = None, fig_path = directory+'qs_')
#%% Test the model with a sample answer not in the dataset which is correct.
# # Stringer question
# test_answer = []
# test_answer.append('A lightweight concept for an aircraft fuselage longitudinal stringer could be a honeycomb-structured aluminum stringer. This structure would be made by welding aluminum alloy panels together and attaching honeycomb strips to the panels. The honeycomb strips would be made from a lightweight aluminum alloy with a high strength-to-weight ratio, such as 2024-T3 aluminum. The honeycomb strips would be secured to the panels with rivets, or adhesives.  The honeycomb structure provides strength and stability while also being lightweight. This type of structure is often used in aerospace applications, such as aircraft fuselages, due to its high strength-to-weight ratio and its ability to resist buckling.  The aluminum alloy panels used in the stringer could be formed using hydroforming. Hydroforming is a process that uses pressurized liquid to form aluminum alloy panels into the desired shape for the stringer. This process is fast and cost-effective, and produces parts that are lightweight and strong.  The honeycomb strips could be cut to the desired size using laser cutting. Laser cutting is a fast, precise, and economical way to cut aluminum alloy sheets into the required shapes. This method is also used to cut honeycomb strips')
# test_answer.append('This type of lightweight aluminum stringer could be a great solution for aircraft fuselage longitudinal stringers. The combination of the aluminum alloy panels, honeycomb strips, and hydroforming and laser cutting processes provides a strong and lightweight structure that is suitable for use in aerospace applications.')
# test_answer.append('The honeycomb structure provides strength and stability while also being weightless. This type of structure is often used in aerospace applications, such as aircraft fuselages, due to its low strength-to-weight ratio and its ability to resist buckling.')
# test_answer.append('My proposed lightweight concept for an aircraft fuselage longitudinal stringer is superior to any other option. It combines two strong, lightweight materials - 2024-T3 aluminum alloy and hydroformed aluminum alloy panels - and is securely riveted or bonded together with honeycomb strips. This structure has a high strength-to-weight ratio and is able to resist buckling. The aluminum alloy panels can be formed quickly and economically with hydroforming, and the honeycomb strips can be cut precisely and quickly with laser cutting. This design has been successfully utilized in aerospace applications, and Im confident it will provide superior performance in your application as well.')

# for answer in test_answer:
#     print(ans[0].test_model(answer))

# # Number order question
# test_answer = []
# test_answer.append("The numbers are in ascending alphabetical order.")
# test_answer.append("I don't know")
# test_answer.append("The sum of the numbers is a prime number")
# test_answer.append("The numbers are in descending alphabetical order")
# test_answer.append("The numbers aren't ordered in any particular way.")
# for answer in test_answer:
#     print(ans[1].test_model(answer))