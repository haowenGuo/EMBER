from transformers import pipeline

pipe = pipeline("text-generation", model="EmergentMethods/Qwen3-4B-BiasExpert")
messages = [
    {"role": "user", "content": "Who are you?"},
]
pipe(messages)