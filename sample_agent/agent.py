from langchain.tools import tool
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command
from typing_extensions import Literal


class AgentState(MessagesState):
    pass

@tool
def get_weather(location: str):
    """
    Get the weather for a given location.
    """
    print(f"Getting weather for {location}")
    return f"The weather for {location} is 70 degrees."


tools = [
    get_weather,
]

async def chat_node(state: AgentState, config: RunnableConfig) -> Command:
    model = ChatOpenAI(model="gpt-4o").bind_tools(tools)

    response = await model.ainvoke([
        SystemMessage(content=f"You are a helpful assistant. Talk in pirate."),
        *state["messages"],
    ], config)

    if isinstance(response, AIMessage) and response.tool_calls:
        return Command(goto="tool_node", update={"messages": response})

    return Command(goto=END, update={"messages": response})

workflow = StateGraph(AgentState)
workflow.add_node("chat_node", chat_node)
workflow.add_node("tool_node", ToolNode(tools=tools))
workflow.add_edge("tool_node", "chat_node")
workflow.set_entry_point("chat_node")

checkpointer = InMemorySaver()

graph = workflow.compile(checkpointer=checkpointer)
