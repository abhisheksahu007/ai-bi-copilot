from openai import AzureOpenAI

client = AzureOpenAI(
    api_key="mUioPGaWXpVytXwabznwffKEXYyRFYTijQcRim2zgGAfoVSkZO23JQQJ99CBACHYHv6XJ3w3AAAAACOGmOy9",
    api_version="2024-02-01",
    azure_endpoint="https://abhis-mlv09ybo-eastus2.cognitiveservices.azure.com/"
)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello AI"}],
)

print(response.choices[0].message.content)