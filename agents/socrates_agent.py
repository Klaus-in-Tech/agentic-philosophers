import os

from dotenv import load_dotenv
from langchain import hub
from langchain.agents import (
    AgentExecutor,
    create_react_agent,
)
from langchain.memory import ConversationBufferMemory
from langchain.schema.output_parser import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

load_dotenv()


def socrates_agent(query: str) -> str:
    llm = ChatOllama(
        temperature=0,
        model="deepseek-r1:1.5b",
    )

    # api_key = os.environ.get("OPENAI_API_KEY")

    # llm = ChatOpenAI(model="gpt-4-turbo-2024-04-09", temperature=0.1,api_key=api_key)

    try:
        with open("PromptTemplates/SocratesAgent.txt") as file:
            scorates_prompt = file.read()
    except Exception:
        print("failed to load socrates prompt")

    # react_prompt = hub.pull("hwchase17/react")

    prompt_template = PromptTemplate(
        input_variables=["user_query"], template=scorates_prompt
    )

    # tools_for_agent = []

    # Initialize memory
    # memory = ConversationBufferMemory(
    #     memory_key="chat_history",
    #     return_messages=True
    # )

    # agent = create_react_agent(llm=llm, tools=tools_for_agent, prompt=react_prompt)

    # agent_executor = AgentExecutor(
    #     agent=agent,
    #     tools=tools_for_agent,
    #     verbose=False,
    #     handle_parsing_errors=True,
    #     max_iterations=4,
    #     # memory=memory
    # )

    chain = prompt_template | llm | StrOutputParser()

    try:
        result = chain.invoke(input={"user_query": query})
    except Exception as e:
        print(f"An exception {e} has occured.")

    return print(result)


if __name__ == "__main__":
    socrates_agent("How can we ensure AI benefits us all.")
