from groq import Groq
import os
import pandas as pd


#read the API key from a file, the tokenGroqAPI.txt file should be in the config folder, if not the script do not work
try:
    with open(os.path.join(os.path.dirname(__file__), 'config/tokenGroqAPI.txt')) as file:
        apy_key = file.read()
except:
    print("The file with the API key was not found")
    os._exit(1)

def read_dataset(path):
    #read the obfuscated dataset
    df = pd.read_csv(path)
    return df

def main():
    #initialize the chat completion   
    # set the model to be used among the list of models
    listOfModels = ["llama2-70b-4096", "mixtral-8x7b-32768", "gemma-7b-it"]
    model = listOfModels[0] #0, 1, 2
    #read the obfuscated dataset
    mechanism = "cmp" #"mhl", "cmp"
    epsilon = 3 #3, 9, 15, 21
    messages = []
    df = read_dataset(f'./data/obfuscated/{mechanism}/{epsilon}.csv', sep = ',', header = 0)
    #iterate over the rows composing new messagges
    for index, row in df.iterrows():
        message = [
            {
                "role": "user",
                "content": f"Respond with True or False: {row['obfuscated_query']}",
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
            max_tokens=10
        )
        #save the response in the dataframe
        df_responses = df_responses.append({'id': index, 
                                            'query': message[0]['content'], 
                                            'response': chat_completion.choices[0].message.content, 
                                            'epsilon': epsilon, 
                                            'model': model, 
                                            'mechanism': mechanism}, 
                                            ignore_index=True)
    #save the dataframe to a file use os to create the output folder if it does not exist and the subfolder with the model name
    if not os.path.exists('./output'):
        os.makedirs('./output')
    if not os.path.exists(f'./output/{model}'):
        os.makedirs(f'./output/{model}')

    df_responses.to_csv(f'./output/{model}/{mechanism}_{epsilon}.csv', index = False)

if __name__ == "__main__":
    main()