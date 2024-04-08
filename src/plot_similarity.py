import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

sns.set_theme(style="whitegrid")
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

plt.rcParams['xtick.labelsize'] = 21
plt.rcParams['ytick.labelsize'] = 21
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.titlesize'] = 21
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['legend.title_fontsize'] = 21


eps = [5, 10, 20, 30, 40, 50]
mechanisms = ['CMP', 'Mahalanobis']

values = []

for mech in mechanisms:
    for epsilon in eps:
        # Load the data
        data = pd.read_csv(f'data/similarity/similarity_{mech}_{epsilon}.csv')
        #get the similarity values
        similarities = data['similarity']
        values.append({'mechanism': mech, 'epsilon': epsilon, 'similarity': similarities.mean()})

df = pd.DataFrame(values)
# Plot the similarity values
fig = plt.figure(figsize=(15, 8))
ax = sns.lineplot(data=df, x='epsilon', y='similarity', hue='mechanism', marker='o', markers=True, style='mechanism', dashes=True, palette='viridis', markersize=10)
#for mahalanobis legend plot x and not o as marker
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles[1:], labels=labels[1:])

ax.set_xlabel(r'$\varepsilon$', fontsize=20)
ax.set_ylabel('Average Cosine Similarity', fontsize=20)
plt.legend(loc='center left', title='Mechanism', bbox_to_anchor=(1.02, 0.8))
plt.tight_layout()
plt.grid(False)
plt.savefig('output/average_similarity.pdf', bbox_inches='tight', format='pdf')
plt.show()
