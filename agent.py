"""
Startup Idea Validator
LangChain Agent + Groq LLM + Tavily Search
Full version (deployment-safe)
"""

# --------------------------------------------------
# IMPORTS
# --------------------------------------------------
import streamlit as st
import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults

# --------------------------------------------------
# LOAD ENV
# --------------------------------------------------
load_dotenv()

# --------------------------------------------------
# STREAMLIT CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="📈 Startup Idea Validator",
    layout="centered"
)

st.title("📈 Startup Idea Validator")
st.write("Agentic AI-based startup evaluation using real market research")

# --------------------------------------------------
# USER INPUTS
# --------------------------------------------------
idea = st.text_area(
    "💡 Startup Idea Description",
    placeholder="Describe your startup idea in detail"
)

target_users = st.text_input(
    "🎯 Target Users",
    placeholder="Students, professionals, businesses, etc."
)

market_info = st.text_area(
    "🌍 Market Information (optional)",
    placeholder="Known competition, assumptions, constraints"
)

analyze_btn = st.button("🚀 Analyze Startup")

# --------------------------------------------------
# TOOLS
# --------------------------------------------------

@tool
def market_research_tool(query: str) -> str:
    """
    Perform live market research using Tavily search.
    """
    tavily = TavilySearchResults(
        max_results=5,
        search_depth="basic"
    )
    results = tavily.invoke(query)

    if not results:
        return "No relevant market data found."

    formatted = []
    for r in results:
        formatted.append(f"- {r.get('content', '')}")

    return "\n".join(formatted)

tools = [market_research_tool]

# --------------------------------------------------
# AGENT CREATION
# --------------------------------------------------
def create_agent():
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.5,
        max_tokens=1200
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are a startup evaluation expert.

Your responsibilities:
- Evaluate startup ideas objectively
- Use market research when competition or trends are unclear
- Avoid assumptions and hallucination
- Base analysis on evidence

You MAY call tools if required.
DO NOT call tools unnecessarily.

Output format MUST be:

Strengths:
- ...

Weaknesses:
- ...

Risk Analysis:
- ...

Monetization Ideas:
- ...

Viability Score (out of 10):
"""
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=5
    )

agent_executor = create_agent()

# --------------------------------------------------
# MEMORY (CHAT HISTORY)
# --------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------
if analyze_btn:

    if not idea.strip() or not target_users.strip():
        st.warning("⚠️ Please fill in required fields.")
    else:
        with st.spinner("🔍 Evaluating startup idea..."):

            user_prompt = f"""
Startup Idea:
{idea}

Target Users:
{target_users}

Market Info:
{market_info if market_info else "Not provided"}

If market competition or trends are unclear, perform market research.
"""

            response = agent_executor.invoke({
                "input": user_prompt,
                "chat_history": st.session_state.chat_history
            })

            # ✅ FIX: extract only final text
            final_output = response.get("output", "No response generated.")

            # update memory
            st.session_state.chat_history.append(
                HumanMessage(content=user_prompt)
            )
            st.session_state.chat_history.append(
                AIMessage(content=final_output)
            )

        st.subheader("📊 Startup Evaluation Result")
        st.markdown(final_output)
