import openai

openai.api_key = open("key.txt", "r").read().strip("\n")

message_history = [{"role": "user",
                    "content": f"You are TennisGPT. You will be given a list of tennis tournaments. Your output will be a csv file. The first column of the table will be the name of each given tennis tournament. The second column will be the 3 letter country code of the associated tournament. The third column will be the city that the associated tournament was played in. The output will be only the csv file, no text beforehand. If the tennis tournament name is not a valid tennis tournament, then include it as a row in the csv file, but leave the country code and city blank. Do not include a header row and only include one line break between each row. If you understand, say OK."},
                   {"role": "assistant", "content": f"OK"}]

input = ['Wimbledon', 'Roland Garros', 'Miami Open', 'US Open', 'teskltjklasfdf']

message_history.append({"role": "user", "content": f"{input}"})

completion = openai.ChatCompletion.create(
    model="gpt-4",  # 10x cheaper than davinci, and better. $0.002 per 1k tokens
    messages=message_history
)

reply_content = completion.choices[0].message.content  # .replace('```python', '<pre>').replace('```', '</pre>')

print((reply_content))
print(repr(reply_content))