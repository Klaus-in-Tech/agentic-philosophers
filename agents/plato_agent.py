from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from chain.conservation_chain import MyConversationChain
from memory import memory
# from clean_llm_output import clean_llm_output

load_dotenv()




def plato_agent(query: str) -> str:
    llm = ChatOllama(
        temperature=0,
        model="mistral:latest",
    )

    # api_key = os.environ.get("OPENAI_API_KEY")

    # llm = ChatOpenAI(model="gpt-4-turbo-2024-04-09", temperature=0.1,api_key=api_key)

    try:
        with open("PromptTemplates/PlatoAgent.txt") as file:
            plato_prompt = file.read()
    except Exception:
        print("Failed to load plato prompt")

    # Plato prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", plato_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )


    chain = MyConversationChain(llm, prompt, memory)

    try:
        result = chain.invoke(query)
    except Exception as e:
        print(f"An exception {e} has occured.")

    chat_history = memory.load_memory_variables({})
    if "chat_history" in chat_history:
           for message in chat_history['chat_history']:
            # Check the type of message and access attributes accordingly
            if isinstance(message, HumanMessage):
                print(f"Human: {message.content}")  # Access content attribute
            elif isinstance(message, AIMessage):
                print(f"AI: {message.content}")  # Access content attribute


    return result
