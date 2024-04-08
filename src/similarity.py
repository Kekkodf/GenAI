from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

sns.set_theme(style="whitegrid")
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

plt.rcParams['xtick.labelsize'] = 21
plt.rcParams['ytick.labelsize'] = 21
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.titlesize'] = 21
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['legend.title_fontsize'] = 21

def cos_sim(v,w):
    return np.dot(v,w) / (np.linalg.norm(v) * np.linalg.norm(w))

#realResponses = pd.read_csv('data/obfuscated/obfuscatedTestRun_CMP_5.csv')
# Keep only unique ids
#realResponses = realResponses.drop_duplicates(subset='question').reset_index(drop=True)

eps = [5, 10, 20, 30, 40, 50]
mechanisms = ['CMP', 'Mahalanobis']

sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')

for mech in mechanisms:
    print(f"Mechanism: {mech}")
    for epsilon in eps:
        print(f"Epsilon: {epsilon}")
        # Load the data
        data = pd.read_csv(f'data/obfuscated/obfuscatedTestRun_{mech}_{epsilon}.csv')
        # Keep only unique ids
        data = data.drop_duplicates(subset='question').reset_index(drop=True)
        columns = ['id', 'question', 'obfuscatedQuestion']
        data = data[columns]
        # Calculate the cosine similarity between the two sentences
        toEncode = data['question'].to_list() + data['obfuscatedQuestion'].to_list()
        embeddings = sbert_model.encode(toEncode)
        similarities = []
        for i in range(len(data)):
            similarities.append(cos_sim(embeddings[i], embeddings[i+len(data)]))
        data['similarity'] = similarities
        #save the data
        if not os.path.exists('data/similarity'):
            os.makedirs('data/similarity')
        data.to_csv(f'data/similarity/similarity_{mech}_{epsilon}.csv', index=False)
        
