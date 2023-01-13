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
generate_embeddings = True
n_clusters = 4  # Determined the number of clusters to use
# Set to none to truly randomize. 42 used in code to reproduce samples to what is in openai docs.
random_state = 10

# Read Data
df = pd.read_csv(directory+file, index_col=0)
df = df[['Question ID', 'Question', 'Answer',
         'Curiousity_grade', 'Hunger_grade', 'Smarts_grade',
         'Curiousity_optimum', 'Hunger_optimum', 'Smarts_optimum']]
df = df.dropna()
metrics = ['Curiousity_grade', 'Hunger_grade', 'Smarts_grade']

#%%Expirement with some vis tools
import seaborn as sns; sns.set()

pair_plots = sns.pairplot(df[df['Question ID'] == 1], height=4, 
                          vars = ['Curiousity_grade', 'Hunger_grade', 'Smarts_grade'],
                          diag_kind = None,
                          kind = 'hist',
                          corner = True);
pair_plots.add_legend(title = 'Legend')

#%% Create open_ended_answer object
question = []
ans = []
for i in range(len(df['Question ID'].unique())):
    q_ID = df['Question ID'].unique()[i]
    question.append(df['Question'][df.index[df['Question ID'] == q_ID].tolist()[0]])
    ans.append(open_ended_tools.OpenEndedAnswer(df[df['Question ID'] == q_ID], metrics))
    
    # ans[i].create_answer_model(directory+file[:-4]+f'_{q_ID}.csv', 
    #                            random_state=random_state, 
    #                            generate_embeddings=generate_embeddings)
    # ans[i].make_clusters(n_clusters=n_clusters, 
    #                      random_state=random_state, 
    #                      ans_per_cluster=1,
    #                      cluster_description_file=directory+file[:-4]+f'_{q_ID}_cd.csv')
    # print(ans[i])
    # ans[i].plot_clusters(random_state=random_state, fig_path=directory+file[:-4]+f'_{q_ID}.png')

#%% Test the model with a sample answer not in the dataset which is correct.

# Stringer question
test_answer = []
# Good long answer from chatGPT
test_answer.append("A minimum weight concept for an aircraft fuselage longitudinal stringer would likely involve the use of lightweight, high-strength materials such as carbon fiber composites. These materials have high specific strength and stiffness, allowing for a thinner, lighter stringer while still maintaining structural integrity.The stringer's shape would be optimized using finite element analysis to minimize weight while meeting strength and stiffness requirements. The optimal shape would likely be a thin, tubular shape with a smooth contour to reduce aerodynamic drag. The manufacturing method could include automated fiber placement or tape laying techniques to create the stringer with minimal material waste. Additionally, advanced curing methods such as out-of-autoclave or vacuum-assisted resin infusion could be used to further reduce the weight of the stringer while maintaining its structural integrity. It's worth noting that, despite no cost or schedule constraints, the final concept and design of the stringer could be a balance between this minimum weight concept and the final assembly, maintenance and performance of the aircraft.")
# Short answer which is pretty bad.
test_answer.append("A minimum weight concept for an aircraft fuselage.")
# Wrong answer.
test_answer.append("Make it out of the same stuff as bridges, steel and lots of I-beams.")
# Correct answer that is not creative
test_answer.append("Use carbon fiber with I, Z, or hat cross sections. To make it as light as possible, cobond or cocure to the skins. You could use fasteners if there are certification reasons to do so. The stringers can be cured on assembly or precured and then attached as described.")
# Odd but correct answer
test_answer.append("3D print the entire structure out of a lightweight alloy such as titanium. The stringer sections should be hat, T, Z, or I shaped.")
for answer in test_answer:
    print(ans[0].test_model(answer))
    
# Number order question
test_answer = []
# Good long answer from chatGPT
test_answer.append("The numbers are in ascending alphabetical order.")
# Short answer which is pretty bad.
test_answer.append("I don't know")
# Wrong answer.
test_answer.append("The sum of the numbers is a prime number")
# Correct answer that is not creative
test_answer.append("The numbers are in descending alphabetical order")
# Odd but correct answer
test_answer.append("The numbers aren't ordered in any particular way.")
for answer in test_answer:
    print(ans[1].test_model(answer))