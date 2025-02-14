from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.agents import (
    create_react_agent,
    AgentExecutor,
)
import os
from langchain.memory import ConversationBufferMemory
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser


load_dotenv()


def plato_agent(query: str) -> str:
    llm = ChatOllama(
        temperature=0,
        model="deepseek-r1:1.5b",
    )

    # api_key = os.environ.get("OPENAI_API_KEY")

    # llm = ChatOpenAI(model="gpt-4-turbo-2024-04-09", temperature=0.1,api_key=api_key)

    try:
        with open("PromptTemplates/PlatoAgent.txt") as file:
            plato_prompt = file.read()
    except Exception:
        print("Failed to load plato prompt")

    # react_prompt = hub.pull("hwchase17/react")

    prompt_template = PromptTemplate(
        input_variables=["user_query"], template=plato_prompt
    )

    # tools_for_agent = []

    # # # Initialize memory
    # # memory = ConversationBufferMemory(
    # #     memory_key="chat_history",
    # #     return_messages=True
    # # )

    # agent = create_react_agent(llm=llm, tools=tools_for_agent, prompt=react_prompt)

    # agent_executor = AgentExecutor(
    #     agent=agent,
    #     tools=tools_for_agent,
    #     verbose=True,
    #     handle_parsing_errors=True,
    #     # max_iterations=4,
    # )

    chain = prompt_template | llm | StrOutputParser()

    try:
        result = chain.invoke(input={"user_query": query})
    except Exception as e:
        print(f"An exception {e} has occured.")

    return print(result)


if __name__ == "__main__":
    plato_agent("How can we ensure AI benefits us all.")
