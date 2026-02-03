import streamlit as st
from typing import Union
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage

# Page configuration
st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="centered")
st.title("🤖 AI Chatbot")

# Initialize session state for conversation history
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Initialize LLM (cached to avoid reloading)
@st.cache_resource
def get_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        api_key=st.secrets.get("GROQ_API_KEY") or None  # Will use env var if not in secrets
    )

# Define the agent state
class AgentState(dict):
    messages: list[Union[HumanMessage, AIMessage]]

# Build the agent graph (cached)
@st.cache_resource
def get_agent():
    llm = get_llm()
    
    def process_input(state: AgentState) -> AgentState:
        response = llm.invoke(state["messages"])
        state["messages"].append(AIMessage(content=response.content))
        return state
    
    graph = StateGraph(AgentState)
    graph.add_node("process_input", process_input)
    graph.add_edge(START, "process_input")
    graph.add_edge("process_input", END)
    return graph.compile()

agent = get_agent()

# Display chat history
for message in st.session_state.conversation_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)
    else:
        with st.chat_message("assistant"):
            st.write(message.content)

# Chat input
if user_input := st.chat_input("Type your message here..."):
    # Add user message to history and display
    st.session_state.conversation_history.append(HumanMessage(content=user_input))
    with st.chat_message("user"):
        st.write(user_input)
    
    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = agent.invoke({"messages": st.session_state.conversation_history})
            st.session_state.conversation_history = result["messages"]
            ai_response = st.session_state.conversation_history[-1].content
            st.write(ai_response)

# Sidebar with options
with st.sidebar:
    st.header("Options")
    if st.button("Clear Chat"):
        st.session_state.conversation_history = []
        st.rerun()
    
    if st.button("Save Conversation"):
        with open("conversation_history.txt", "w") as f:
            for message in st.session_state.conversation_history:
                role = "User" if isinstance(message, HumanMessage) else "AI"
                f.write(f"{role}: {message.content}\n")
                if isinstance(message, AIMessage):
                    f.write("\n")
        st.success("Conversation saved!")
