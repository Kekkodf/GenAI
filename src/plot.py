import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16

realResponses = pd.read_csv('data/obfuscated/obfuscatedTestRun_CMP_5.csv')
# Keep only unique ids
realResponses = realResponses.drop_duplicates(subset='question').reset_index(drop=True)

eps = [5, 10, 20, 30, 40, 50]
models = ['llama2-70b-4096', 'mixtral-8x7b-32768', 'gemma-7b-it']

# Initialize a dictionary to store average precision values for each model and mechanism
avg_precision_cmp = {model: [] for model in models}
avg_precision_mahalanobis = {model: [] for model in models}

df = pd.DataFrame(columns=["model", "mechanism", "epsilon", "precision"])
for model in models:
    for e in eps:
        mechanism = 'CMP'
        path_cmp = f'output/{model}/{mechanism}/{mechanism}_{e}.csv'
        df_cmp = pd.read_csv(path_cmp)
        responses_cmp = df_cmp['response']
        # Compare the responses with the real responses for CMP mechanism
        realResponses['response_cmp'] = responses_cmp
        realResponses['correct_cmp'] = realResponses.apply(lambda x: x['response_cmp'] == x['answer'], axis=1)
        # Compute precision (accuracy) for CMP mechanism
        precision_cmp = realResponses['correct_cmp'].mean()
        avg_precision_cmp[model].append(precision_cmp)

        temp = pd.DataFrame({
            "model": [model],
            "mechanism": [mechanism],
            "epsilon": [e],
            "precision": [precision_cmp]
        })
        # Append the results to the dataframe
        df = pd.concat([df, temp], ignore_index=True)

        mechanism = 'Mahalanobis'
        path_mahalanobis = f'output/{model}/{mechanism}/{mechanism}_{e}.csv'
        df_mahalanobis = pd.read_csv(path_mahalanobis)
        responses_mahalanobis = df_mahalanobis['response']
        # Compare the responses with the real responses for Mahalanobis mechanism
        realResponses['response_mahalanobis'] = responses_mahalanobis
        realResponses['correct_mahalanobis'] = realResponses.apply(lambda x: x['response_mahalanobis'] == x['answer'], axis=1)
        # Compute precision (accuracy) for Mahalanobis mechanism
        precision_mahalanobis = realResponses['correct_mahalanobis'].mean()
        avg_precision_mahalanobis[model].append(precision_mahalanobis)

        temp = pd.DataFrame({
            "model": [model],
            "mechanism": [mechanism],
            "epsilon": [e],
            "precision": [precision_mahalanobis]
        })
        # Append the results to the dataframe
        df = pd.concat([df, temp], ignore_index=True)

df['model'] = df['model'].map({'mixtral-8x7b-32768': 'Mixtral', 
                               'gemma-7b-it': 'Gemma', 
                               'llama2-70b-4096': 'Llama2'})

# Save the results to a CSV file
df.to_csv('output/average_precision.csv', index=False)

# Plot the results

fig = plt.figure(figsize=(15, 8))
ax = sns.lineplot(x='epsilon', y='precision', hue='model', style='mechanism', data=df, markers=True, markersize=10)

h, l = ax.get_legend_handles_labels()
l1 = plt.legend(h[1:4], l[1:4], title='Model', loc='center left', bbox_to_anchor=(1.02, 0.8))
l2 = plt.legend(h[5:], l[5:], title='Mechanism', loc='center left', bbox_to_anchor=(1.02, 0.6))
ax.add_artist(l1)
ax.set_xlabel(r'$\varepsilon$', fontsize=20)
ax.set_ylabel('Average Precision', fontsize=20)
plt.tight_layout()
plt.grid(False)
plt.savefig('output/average_precision_plot.pdf', bbox_inches='tight', format='pdf')
plt.show()

