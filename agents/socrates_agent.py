from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_ollama import ChatOllama
from memory import memory
from chain.conservation_chain import MyConversationChain

# from clean_llm_output import clean_llm_output

load_dotenv()


def socrates_agent(query: str) -> str:
    llm = ChatOllama(
        temperature=0,
        model="mistral:latest",
    )

    # api_key = os.environ.get("OPENAI_API_KEY")

    # llm = ChatOpenAI(model="gpt-4-turbo-2024-04-09", temperature=0.1,api_key=api_key)

    try:
        with open("PromptTemplates/SocratesAgent.txt") as file:
            socrates_prompt = file.read()
    except Exception:
        print("Failed to load plato prompt")

    # Plato prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", socrates_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    chain = MyConversationChain(llm, prompt, memory)

    try:
        result = chain.invoke(query)
    except Exception as e:
        print(f"An exception {e} has occured.")

    return result
