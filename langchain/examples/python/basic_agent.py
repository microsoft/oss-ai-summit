"""
LangChain Agent With Azure AI init_chat_model and 
OpenAI v1 with Azure AI Endpoint.
"""
import os

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

load_dotenv()

def azure_ai() -> str:
    # pip install langchain-azure-ai

    os.environ["AZURE_AI_ENDPOINT"] = os.getenv("AZURE_AI_ENDPOINT", "")
    os.environ["AZURE_AI_CREDENTIAL"] = os.getenv("AZURE_AI_CREDENTIAL", "")

    def get_weather(city: str) -> str:
        """Get weather for a given city."""
        return f"It's always sunny in {city}!"
    

    agent = create_agent(
        model="azure_ai:gpt-5-mini",
        tools=[get_weather],
        system_prompt="You are a helpful assistant",
    )

    # Run the agent
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
    )

    return response['messages'][-1].content



def open_ai_v1():
    # pip install langchain-openai
    
    # "https://{your_resource_name}.openai.azure.com/openai/v1/",
    resource_name = os.getenv("AZURE_OPEN_AI_RESOURCE_NAME", "your_resource_name")
    base_url = f"https://{resource_name}.openai.azure.com/openai/v1/"

    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
    )

    def addition_tool(a: int, b: int) -> str:
        """Add two numbers."""
        return str(a + b)


    llm = ChatOpenAI(
    model = "gpt-4.1-mini",
    base_url = base_url,
    api_key = token_provider
    )


    agent = create_agent(
        model=llm,
        tools=[addition_tool],
        system_prompt="You are a helpful assistant",
    )

    # Run the agent with streaming
    final_message = None

    for chunk in agent.stream(  
        {"messages": [{"role": "user", "content": "add 2 + 2"}]},
        stream_mode="updates",
    ):
        for step, data in chunk.items():
            print(f"step: {step}")
            print(f"content: {data['messages'][-1].content_blocks}")
            # Capture the final message
            final_message = data['messages'][-1].content
    
    return final_message



if __name__ == "__main__":
    # result_1 = azure_ai()
    # print(result_1)
    result_2 = open_ai_v1()
    print("")
    print(f"final answer: {result_2}")
    
