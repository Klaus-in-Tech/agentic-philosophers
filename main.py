import streamlit as st
from langchain_ollama import ChatOllama
from philosophical_debate import PhilosophicalDebate


def main():
    st.header("🦜🔗 Agentic Philosophers App")


    user_query = st.text_input("Prompt", placeholder="Enter your prompt here:")
    debate = PhilosophicalDebate()

    if user_query:
        with st.spinner("Thinking"):
            generated_response = debate.conduct_debate(initial_question=user_query)
            st.write(generated_response)


if __name__ == "__main__":
    main()