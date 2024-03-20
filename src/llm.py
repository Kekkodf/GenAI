from groq import Groq

with open('./config/tokenGroqAPI.txt', 'r') as file:
    apy_key = file.read()

def main():
    client = Groq(
        api_key= apy_key, 
        )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "Explain the importance of low latency LLMs",
            }
        ],
        model="mixtral-8x7b-32768",
    )

    print(chat_completion.choices[0].message.content)

    #save the response to a file
    with open('./output/llm.txt', 'w') as file:
        file.write(chat_completion.choices[0].message.content)

if __name__ == "__main__":
    main()