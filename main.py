from langchain_ollama import ChatOllama


llm = ChatOllama(
    temperature=0,
    model="deepseek-r1:1.5b",
)

try:
    with open("PromptTemplates/assistantPrompt.txt") as file:
        assistant_prompt = file.read()
except Exception:
    print("failed to load Update system propmt")
