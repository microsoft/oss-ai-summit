import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()


def send_email_tool(recipient: str, subject: str, body: str) -> str:
    """Simulate sending an email."""
    return f"Email sent to {recipient} with subject '{subject}'."

def read_email_tool(email_id: str) -> str:
    """Simulate reading an email."""
    return f"Contents of email {email_id}: Hello, this is a test email."

os.environ["AZURE_AI_ENDPOINT"] = os.getenv("AZURE_AI_ENDPOINT", "")
os.environ["AZURE_AI_CREDENTIAL"] = os.getenv("AZURE_AI_CREDENTIAL", "")

agent = create_agent(
    model="azure_ai:gpt-5-mini",
    tools=[read_email_tool, send_email_tool],
    checkpointer=InMemorySaver(),
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                # Require approval, editing, or rejection for sending emails
                "send_email_tool": {
                    "allowed_decisions": ["approve", "edit", "reject"],
                },
                # Auto-approve reading emails
                "read_email_tool": False,
            }
        ),
    ],
)

config = {"configurable": {"thread_id": "1"}}

# Use stream to handle interrupts
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": """
                   Please send an email to alice@example.com with subject 'Meeting' and body 
                   'Let's meet at 10 AM.'"""}]},
    config=config,
    stream_mode="updates"
):
    print(f"\nChunk: {chunk}")
    
    # Check if there's an interrupt
    if "__interrupt__" in chunk:
        interrupt_data = chunk["__interrupt__"]
        print("\n🛑 INTERRUPT DETECTED!")
        print(f"Tool: {interrupt_data}")
        
        # Handle the interrupt - you can prompt user for input here
        # For now, we'll auto-approve
        decision = "approve"  # Change to input("Decision (approve/edit/reject): ") for interactive mode
        
        if decision == "approve":
            # Continue execution with approval
            for continuation in agent.stream(
                None,  # No new input needed
                config=config,
                stream_mode="updates"
            ):
                print(f"\nContinuation: {continuation}")
        elif decision == "reject":
            print("Tool execution rejected!")
            break

print("\n✅ Done!")