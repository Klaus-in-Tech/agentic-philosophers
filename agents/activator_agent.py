from langchain_core.prompts.prompt import PromptTemplate
from agents.aristotle_agent import aristotle_agent
from agents.plato_agent import plato_agent
from agents.socrates_agent import socrates_agent
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


def activator_agent(self, query):
    # Initialize agent chains
    self.socrates_chain = (
        RunnablePassthrough() | (lambda x: socrates_agent(x)) | StrOutputParser()
    )
    self.plato_chain = (
        RunnablePassthrough() | (lambda x: plato_agent(x)) | StrOutputParser()
    )
    self.aristotle_chain = (
        RunnablePassthrough() | (lambda x: aristotle_agent(x)) | StrOutputParser()
    )

    """Load and return the debate format from file"""
    try:
        with open("PromptTemplates/DebateFormat.txt") as file:
            debate_format = file.read()
        return debate_format
    except Exception as e:
        print(f"\033[91mFailed to load debate format due to : {e}\033[0m")
        return ""

    prompt_template = PromptTemplate(
        template=debate_format,
        input_variables=[
            "SocratesName",
            "PlatoName",
            "AristotleName",
            "chat_history",
        ],  # These are required by ReAct
    )

    # Format the prompt with philosopher names and chat history
    formatted_prompt = self.prompt_template.format(
        SocratesName="Socrates",
        PlatoName="Plato",
        AristotleName="Aristotle",
        chat_history=chat_history,
    )

    if "Socrates" in chat_history:
        self.socrates_chain.invoke(query)
    elif "Plato" in chat_history:
        self.plato_chain.invoke(chat_history)
    elif "Aristotle" in chat_history:
        self.aristotle_chain.invoke(chat_history)

    return
