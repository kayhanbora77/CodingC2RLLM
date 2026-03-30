import openai

client = openai.OpenAI(
    base_url="https://api.llm7.io/v1",
    api_key="unused",  # No key needed
)

response = client.chat.completions.create(
    model="gpt-4o-mini-2024-07-18",
    messages=[
        {"role": "user", "content": "Tell me a short story about a brave squirrel."}
    ],
)

print(response.choices[0].message.content)
