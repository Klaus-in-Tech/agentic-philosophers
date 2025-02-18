from operator import itemgetter
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, Runnable
from langchain_core.output_parsers import StrOutputParser


# Let's build our own ConversationChain!
class MyConversationChain(Runnable):
    def __init__(self, llm, prompt, memory, input_key="input"):
        self.prompt = prompt
        self.memory = memory
        self.input_key = input_key

        # Let's try chaining using LCEL!
        self.chain = (
            RunnablePassthrough.assign(
                chat_history=RunnableLambda(self.memory.load_memory_variables)
                | itemgetter(memory.memory_key)
            )
            | prompt
            | llm
            | StrOutputParser()
        )

    def invoke(self, query, configs=None, **kwargs):
        answer = self.chain.invoke({self.input_key: query})
        self.memory.save_context(
            inputs={"human": query}, outputs={"ai": answer}
        )  # Store the conversation history directly in the memory.
        return answer
