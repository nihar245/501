from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)
prompt_template = ChatPromptTemplate(messages=[
    ("system", "You are a helpful {topic} assistant."),
    ("user", "Tell {number} things about {topic}.")
])

def demo_prompt_template():
    p = prompt_template.invoke({"topic": "AI", "number": "3"})
    print("\nPromptTemplate demo:")
    for m in p.messages:
        print(f"{m.type}: {m.content}")

def demo_simple_invoke():
    print("\nSimple invoke (string):")
    ans = llm.invoke("What is your name?")
    print(ans.content)

def demo_message_invokes():
    print("\nMessage invoke (single turn):")
    msgs = [SystemMessage(content="You are a good assistant in Mathematics."), HumanMessage(content="What is 5 multiplied by 9?")]
    ans1 = llm.invoke(msgs)
    print(ans1.content)

    print("\nFollow-up using history:")
    history = [
        SystemMessage(content="You are a good assistant in Mathematics."),
        HumanMessage(content="What is 5 multiplied by 9?"),
        AIMessage(content=ans1.content),
        HumanMessage(content="If one of them is negative then?")
    ]
    ans2 = llm.invoke(history)
    print(ans2.content)

def demo_repl():
    history = [SystemMessage(content="You are a helpful assistant.")]
    while True:
        q = input("\nYour query (type 'exit' to quit): ").strip()
        if q.lower() == "exit":
            break
        history.append(HumanMessage(content=q))
        r = llm.invoke(history)
        print(r.content)
        history.append(AIMessage(content=r.content))

if __name__ == "__main__":
    demo_prompt_template()
    demo_simple_invoke()
    demo_message_invokes()
    run = input("\nRun interactive REPL? (y/N): ").strip().lower()
    if run == "y":
        demo_repl()