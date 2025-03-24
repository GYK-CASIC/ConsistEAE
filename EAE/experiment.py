import json
from openai import OpenAI
import openai

client = OpenAI(
    api_key="",
    base_url="https://api.agicto.cn/v1"
)

few_shot = 1
messages = []  # Store messages
response = []  # Store model responses

# Load prompts
data_file = open(f"EAE/prompt.json", 'r', encoding="utf-8")
lines = json.load(data_file)

# Load sentences for matching with model-generated responses
sentence_file = open("EAE/sentences.json", 'r', encoding="utf-8")

# Iterate through all prompts in `lines` and add them as messages with the "user" role
for i in range(len(lines)):
    messages.append({"role": "user", "content": lines[i]})

response_file = open(f"EAE/response.json", "w", encoding="utf-8")

# Interact with the model
for i in range(len(messages)):
    chat_completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[messages[i]],
        temperature=0.3,
        top_p=0.5
    )
    # Get the assistant's response
    assistant_reply = chat_completion.choices[0].message.content
    print("Response:", assistant_reply)

    # Store the assistant's response
    response.append({"sentence": sentence[i], "content": assistant_reply})

json.dump(response, response_file, ensure_ascii=False, indent=2)
response_file.write("\n")

'''
response_file = open("response0317.json/response_EEA_with_trigger_5-shot_2.json", "w", encoding="utf-8")

# Interact with the model
for i in range(2000, len(messages)):
    chat_completion = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[messages[i]],
        temperature=0.2,
        top_p=0.1
    )
    # Get the assistant's response
    assistant_reply = chat_completion.choices[0].message.content
    print("Response:", assistant_reply)

    # Store the assistant's response
    response0317.json.append({"sentence": sentence[i], "content": assistant_reply})

json.dump(response0317.json, response_file, ensure_ascii=False, indent=2)
response_file.write("\n")
'''
