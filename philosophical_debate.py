from typing import Dict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from agents.aristotle_agent import aristotle_agent
from agents.plato_agent import plato_agent
from agents.socrates_agent import socrates_agent


class PhilosophicalDebate:
    def __init__(self):
        # Create callable chains from your existing agent functions
        self.socrates_chain = (
            RunnablePassthrough() | (lambda x: socrates_agent(x)) | StrOutputParser()
        )
        self.plato_chain = (
            RunnablePassthrough() | (lambda x: plato_agent(x)) | StrOutputParser()
        )
        self.aristotle_chain = (
            RunnablePassthrough() | (lambda x: aristotle_agent(x)) | StrOutputParser()
        )

        # Combine them into a sequential chain
        self.debate_chain = (
            self.socrates_chain | self.plato_chain | self.aristotle_chain
        )

    def conduct_debate(self, initial_question: str) -> Dict[str, str]:
        """
        Conduct a philosophical debate using the three agents.
        """
        print(f"\n=== Starting Debate on: {initial_question} ===\n")

        try:
            # First, Socrates responds to user input
            socrates_response = self.socrates_chain.invoke(initial_question)
            # print(f"Socrates: {socrates_response}\n")

            # Then Plato responds to Socrates
            plato_response = self.plato_chain.invoke(socrates_response)
            # print(f"Plato: {plato_response}\n")

            # Then, Aristotle responds to Socrates
            artistotle_response = self.aristotle_chain.invoke(socrates_response)
            print(f"Aristotle: {aristotle_agent}\n")

            # Finally, Socrates provides synthesis
            final_response = self.socrates_chain.invoke("Based on plato's response "+plato_response+" and "+"Artistotle's response  "+artistotle_response)
            # print(f"Socrates: {final_response}\n")

            debate_result = {
                "question": initial_question,
                "socrates_response": socrates_response,
                "plato_response": plato_response,
                "aristotle_response": artistotle_response,
                "final_response":final_response,
            }

            return debate_result

        except Exception as e:
            print(f"An error occurred during the debate: {str(e)}")
            return None


def main():
    debate = PhilosophicalDebate()
    question = "How can we ensure AI benefits all of humanity?"
    result = debate.conduct_debate(question)

    if result:
        print("\n=== Debate Summary ===")
        print(f"Question: {result['question']}")
        print(f"Socrates' View: {result['socrates_response']}...")
        print(f"Plato's Response: {result['plato_response']}...")
        print(f"Aristotle's Synthesis: {result['aristotle_response']}...")
        print(f"Scorates's Synthesis: {result['final_response']}...")



if __name__ == "__main__":
    main()