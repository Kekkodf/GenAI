import json
import pandas
    
def main():
    path = './data/boolQ/'
    #open the json file using pandas
    with open(path+'train.jsonl') as f:
        json_list = list(f)
    data=[]
    for json_str in json_list:
        result = json.loads(json_str)
        data.append(result)
    #create a dataframe
    df = pandas.DataFrame()
    #save the data in the dataframe
    df['question'] = [d['question'] for d in data]
    df['answer'] = [d['answer'] for d in data]
    #lowercaser query
    df['question'] = df['question'].str.lower()
    #remove punctuation
    df = df.replace('[^\w\s]','', regex=True)
    #drop duplicates
    df = df.drop_duplicates('question')
    
    #considering the answer column, keep 50 rows with answer 'yes' and 50 rows with answer 'no'
    df = df.groupby('answer').head(25)
    #shuffle the dataframe
    df = df.sample(frac=1).reset_index(drop=True)
    #add column with id
    df.insert(0, 'id', range(1, 1 + len(df)))
    #save into a csv file
    df.to_csv(path+'exam.csv', index=False)
    #save the final dataframe
    #df.to_csv(path+'boolQ.csv', index=False)
    #print(df.info())

if __name__ == '__main__':
    main()
