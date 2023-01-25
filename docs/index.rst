.. engineer_tests documentation master file, created by
   sphinx-quickstart on Wed Jan 25 08:00:24 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to engineer_tests's documentation!
==========================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:


# engineer_tests
Code resposity for grading of important metrics for open ended questions in aerospace engineer hardware development engineer tests. Current metrics being evaluated are:
* Curiousity
* Hunger
* Smarts

The code has two methods of evaluating these metrics for a question/answer set, each of which are based on the use of embeddings from openai (https://beta.openai.com/docs/guides/embeddings/use-cases).
* **Method 1**: Embeddings as a feature encoder for a machine learning algorithm. This requires a dataset which has been pregraded for the desired metrics. Kmeans clustering is also used for visualization and can be helpful to populate the training set and test new answers. Works reasonably well
* **Method 2**: Zero-shot classification with openai embedding cosine distance. This method does not work well with no fine tuning of the openai default model. 

## Inputs
* **Training dataset**: Works with a set of questions and answers with labeled data in a csv format. There is a CSV file located in the drive folder under relevant sources, which contains a dataset which has been used to train and provides reasonable predictability (around +-0.2 mean squared error on a range of -1 to 1)
* **Metric list**: A list of metrics with category terms. See the top description for meain metrics. Category terms could be as follows for Curiousity: Creativity, Innovation, Curiousity.
* **API key for openai**: https://github.com/openai/openai-python#usage

## Run scripts
* **generate_new_answers.py**: Uses tuned openai davinci model parameters (https://beta.openai.com/playground/p/va8KFZYqyjihR4hoYKzU1OiY?model=text-davinci-003) to give highly creative and random answers to open ended questions. Use this to add to the dataset in method 1. You will have to modify the answers, since the outputs are a combination of genius insights and complete garbage.
* **grade_open_ended.py**: Reads CSV files and performs grading on metrics based on method 1 and 2 above. 

## Outputs
See run scripts

# Relevant sources
Drive folder: https://drive.google.com/drive/folders/1BsTPHNdEywMpFs_wwvioMqscgJuMu81K?usp=share_link

Modified

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
