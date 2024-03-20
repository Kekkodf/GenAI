#read data.txt
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
sns.set_color_codes("colorblind")

def read_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def main():
    try:
        dataset_name = ['s08', 's09', 's10']
        for name in dataset_name:
            data = read_file('./data/questionAnswer/'+name+'.txt')
            #save data in a csv file using the first row as the header
            data = data.split('\n')
            data = [i.split('\t') for i in data]
            df = pd.DataFrame(data[1:], columns=data[0])
            #drop columns ArticleTitle,Question,Answer,DifficultyFromQuestioner,DifficultyFromAnswerer,ArticleFile
            df = df.drop(columns=['ArticleTitle', 'DifficultyFromQuestioner', 'DifficultyFromAnswerer', 'ArticleFile'])
            #drop duplicates
            df = df.drop_duplicates('Question')
            #lowercaser all columns
            df = df.apply(lambda x: x.str.lower())
            #remove punctuation
            df = df.replace('[^\w\s]','', regex=True)
            #keep only answers that are yes or no
            df = df[df['Answer'].isin(['Yes', 'No', 'yes', 'no'])]
            df.to_csv(f'./data/questionAnswer/{name}.csv', index=False)

        #concatenate all dataframes
        df = pd.concat([pd.read_csv('./data/questionAnswer/'+name+'.csv') for name in dataset_name])
        #drop duplicates
        df = df.drop_duplicates('Question')
        #add column with id
        df.insert(0, 'id', range(1, 1 + len(df)))
        #substitute yes for True and no for False as boolean values
        df['Answer'] = df['Answer'].map({'Yes': True, 'yes': True, 'No': False, 'no': False})
        #save the final dataframe
        df.to_csv('./data/questionAnswer/Q&A.csv', index=False)
        for name in dataset_name:
            os.remove('./data/questionAnswer/'+name+'.csv')

        #plot a histogram of the answers
        #df = pd.read_csv('./data/questionAnswer_dataset/dataset_Q&A.csv')
        #df['Answer'].value_counts().plot(kind='bar')
        #plt.show()
    except:
        pass

if __name__ == '__main__':
    main()