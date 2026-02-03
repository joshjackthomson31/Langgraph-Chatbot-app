from typing import Dict, TypedDict, Union
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage

class AgentState(TypedDict):
    messages: list[Union[HumanMessage, AIMessage]]

llm = ChatOllama(model="llama3", temperature=0)

def process_input(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    print(f"AI: {response.content}")
    state["messages"].append(AIMessage(content=response.content))
    return state

graph = StateGraph(AgentState)
graph.add_node("process_input", process_input)
graph.add_edge(START, "process_input")
graph.add_edge("process_input", END)
agent = graph.compile()

conversation_history = []

user_input = input("Enter your message: ")
while user_input != "exit":
    conversation_history.append(HumanMessage(content=user_input))
    result = agent.invoke({"messages": conversation_history})
    conversation_history = result["messages"]
    user_input = input("Enter your message: ")

with open("conversation_history.txt", "w") as f:
    for i, message in enumerate(conversation_history):
        role = "User" if isinstance(message, HumanMessage) else "AI"
        f.write(f"{role}: {message.content}\n")
        # Add a blank line after each AI message (end of a conversation exchange)
        if isinstance(message, AIMessage):
            f.write("\n")