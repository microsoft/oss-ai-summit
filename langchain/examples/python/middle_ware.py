import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

load_dotenv()

os.environ["AZURE_AI_ENDPOINT"] = "https://models.github.ai/inference"
os.environ["AZURE_AI_CREDENTIAL"] = os.environ["GITHUB_TOKEN"]

def send_email_tool(recipient: str, subject: str, body: str) -> str:
    """Simulate sending an email."""
    return f"Email sent to {recipient} with subject '{subject}'."

def read_email_tool(email_id: str) -> str:
    """Simulate reading an email."""
    return f"Contents of email {email_id}: Hello, this is a test email."


agent = create_agent(
    model="azure_ai:gpt-4.1",
    tools=[read_email_tool, send_email_tool],
    checkpointer=InMemorySaver(),
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_email_tool": True,  # Require approval for sending emails
                "read_email_tool": False,  # Auto-approve reading emails
            }
        ),
    ],
)

config = {"configurable": {"thread_id": "1"}}

# First invoke - will pause at the interrupt
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Please send an email to yohan@example.com with subject 'Meeting' and body 'Let's meet at 10 AM.'"}]},
    config=config
)

print(f"🤖 Agent: {result['messages'][-1].content}")

# Check if we have an interrupt
state = agent.get_state(config)
if state.tasks:
    print("\n🛑 INTERRUPT - Human approval required!")
    for task in state.tasks:
        if hasattr(task, 'interrupts'):
            for interrupt in task.interrupts:
                print(f"Details: {interrupt.value}")
    
    # Get user decision
    decision = input("\nApprove? (y/n): ").strip().lower()
    
    if decision == 'y':
        # Resume with approval
        result = agent.invoke(
            Command(resume={"decisions": [{"type": "approve"}]}),
            config=config
        )
        print(f"\n🤖 Agent: {result['messages'][-1].content}")
    else:
        print("❌ Rejected!")

print("\n✅ Done!")
