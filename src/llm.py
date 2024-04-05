from groq import Groq
import os
import pandas as pd
from tqdm import tqdm
import time

#read the API key from a file, the tokenGroqAPI.txt file should be in the config folder, if not the script do not work
try:
    with open('./config/tokenGroqAPI.txt') as file:
        apy_key = file.read()
except:
    print("The file with the API key was not found")
    os._exit(1)

def read_dataset(path):
    #read the obfuscated dataset
    #the dataset should have the following columns: <id,question,answer,obfuscatedQuestion,mechanism>
    df = pd.read_csv(path, sep=',', header=0, encoding='utf-8')
    return df

def main():
    #all available models
    listOfModels = ["llama2-70b-4096", "mixtral-8x7b-32768", "gemma-7b-it"]
    #setting the epsilons to be used
    epsilons = [5, 10, 20, 30, 40, 50]
    #and the mechanisms
    mechanisms = ["CMP", "Mahalanobis"]
    #iterate over the models, mechanisms and epsilons
    for model in listOfModels:
        print(f"Model: {model}")
        for mechanism in mechanisms:
            print(f"Mechanism: {mechanism}")
            for epsilon in epsilons:
                print(f"Epsilon: {epsilon}")
                t_0 = time.time()
                #initialize the chat completion   
                messages = []
                df = read_dataset(f'./data/obfuscated/obfuscatedTestRun_{mechanism}_{epsilon}.csv')
                #sample one row for each id
                df = df.groupby('id').sample(1, random_state=42).reset_index(drop=True)

                #composing new messagges
                for index, row in df.iterrows():
                    message = [
                        {
                            "role": "user",
                            "content": f"Limit your response to True or False in answering the following question (do not repeat the question in the response): {row['obfuscatedQuestion']}",
                        }
                    ]
                #add the messages to the list of messages
                    messages.append(message)

                #initialize the dataframe to save the responses
                df_responses = pd.DataFrame(columns=["id", "query", "response", "epsilon", "model", "mechanism"])

                #initialize client with Groq API key
                client = Groq(
                    api_key= apy_key, 
                    )
                for index, message in enumerate(messages):
                    chat_completion = client.chat.completions.create(
                        messages=message,
                        #set the model to be used
                        model=model, #llama2-70b-4096, mixtral-8x7b-32768, gemma-7b-it
                        #set the max tokens
                        max_tokens=20
                    )
                    #save the response in a dict to save in final dataset
                    resp = {'id': index,
                            'query': df['question'].iloc[index],
                            'response': chat_completion.choices[0].message.content,
                            'epsilon': epsilon,
                            'model': model,
                            'mechanism': mechanism}
                    #insert the response in the dataframe
                    temp = pd.DataFrame([resp], columns=["id", "query", "response", "epsilon", "model", "mechanism"])
                    df_responses = pd.concat([df_responses, temp], ignore_index=True)

                #save the dataframe
                if not os.path.exists('./output'):
                    os.makedirs('./output')
                if not os.path.exists(f'./output/{model}'):
                    os.makedirs(f'./output/{model}')
                if not os.path.exists(f'./output/{model}/{mechanism}'):
                    os.makedirs(f'./output/{model}/{mechanism}')

                df_responses.to_csv(f'./output/{model}/{mechanism}/{mechanism}_{epsilon}.csv', index = False)

                #get info about performance
                t_1 = time.time()
                print(f"Time elapsed: {t_1 - t_0}")

if __name__ == "__main__":
    main()