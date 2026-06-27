from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import add_messages
from langgraph.checkpoint.memory import InMemorySaver
 
load_dotenv()

llm = ChatGroq(
    model='llama-3.1-8b-instant', temperature=0.8,
)


class ChatState(TypedDict):
    messages : Annotated[list[BaseMessage],add_messages]


def input_processing(state:ChatState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {'messages': [response]}

checkpointer = InMemorySaver()

graph = StateGraph(ChatState)

graph.add_node("Input_processing",input_processing)

graph.add_edge(START, "Input_processing")
graph.add_edge("Input_processing", END)

chatbot= graph.compile(checkpointer = checkpointer)




    