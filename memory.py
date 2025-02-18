# memory.py
from langchain.memory import ConversationBufferMemory

# Define the memory object globally
memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")
