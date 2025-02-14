from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from langchain_core.prompts.prompt import PromptTemplate
from langchain.agents import (
    create_react_agent,
    AgentExecutor,
)
import os
from langchain.memory import ConversationBufferMemory
from langchain import hub
from langchain_openai import ChatOpenAI
from agents.tools.tools import get_latest_info
from langchain_core.tools import Tool


load_dotenv()


def aristotle_agent(query: str) -> str:
    llm = ChatOllama(
        temperature=0,
        model="deepseek-r1:1.5b",
    )

    # api_key = os.environ.get("OPENAI_API_KEY")

    # llm = ChatOpenAI(model="gpt-4-turbo-2024-04-09", temperature=0.1,api_key=api_key)

    try:
        with open("PromptTemplates/AristotleAgent.txt") as file:
            aristotle_prompt = file.read()
    except Exception:
        print("failed to load socrates prompt")

    react_prompt = hub.pull("hwchase17/react")

    prompt_template = PromptTemplate(
        template=aristotle_prompt,
        input_variables=["user_query"],  # These are required by ReAct
    )

    tools_for_agent = [
        Tool(
            name="Crawl Google 4 lastest info",
            func=get_latest_info,
            description="Useful for getting latest information. Input should be a specific search query string.",
        )
    ]
    # Initialize memory
    # memory = ConversationBufferMemory(
    #     memory_key="chat_history",
    #     return_messages=True
    # )

    agent = create_react_agent(llm=llm, tools=tools_for_agent, prompt=react_prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools_for_agent,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=4,
    )

    result = agent_executor.invoke(
        input={"input": prompt_template.format_prompt(user_query=query)}
    )

    return result["output"]


if __name__ == "__main__":
    aristotle_agent("How can we ensure AI benefits all of humanity.")
