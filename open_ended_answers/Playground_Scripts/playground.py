# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 22:46:47 2022

@author: dsmue
openai.api_key='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'
https://openai.com/blog/introducing-text-and-code-embeddings/
"""

import openai, numpy as np

# Describe an optimal structure for a conventional airplane wing
# Sensible
ans=["The optimal structure for a conventional airplane wing should include a strong, lightweight airframe made of aluminum alloy, a single main spar to provide support and rigidity, and a cantilever design for the wings which will allow them to be self-supporting."]
ans.append("The optimal structure for a conventional airplane wing should include a stressed skin design, a three-dimensional airfoil shape to reduce drag and increase lift, and a composite material construction to reduce weight and increase strength.")

# Borderline
ans.append("Spars and ribs and skins, it totally does not matter where.")
ans.append("Lightweight aerodynamically efficient")

# Incorrect
ans.append("A truss bridge is an optimal structure for a bridge as it is typically lightweight, requires less material than other bridge structures, and is strong enough to support the weight of traffic crossing it.")
ans.append("A cable-stayed bridge is also an optimal structure for a bridge as it has the ability to span large distances and the cables provide great stability and support.")

resp = openai.Embedding.create(
    input=ans,
    engine="text-similarity-davinci-001")

embedding=[]
for i in range(len(ans)):
    embedding.append(resp['data'][i]['embedding'])

similarity_score_00 = np.dot(embedding[0], embedding[0])
similarity_score_01 = np.dot(embedding[0], embedding[1])
similarity_score_02 = np.dot(embedding[0], embedding[2])
similarity_score_03 = np.dot(embedding[0], embedding[3])
similarity_score_04 = np.dot(embedding[0], embedding[4])
similarity_score_05 = np.dot(embedding[0], embedding[5])

# print(embedding[0])
print(similarity_score_00)
print(similarity_score_01)
print(similarity_score_02)
print(similarity_score_03)
print(similarity_score_04)
print(similarity_score_05)