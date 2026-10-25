from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="distilgpt2"
)

response = pipe(
    "Explain law in very simple words",
    max_new_tokens=50
)

print(response[0]["generated_text"])
