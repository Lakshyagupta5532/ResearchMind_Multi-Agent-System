from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools import web_search, scrape_url
import os
from dotenv import load_dotenv


load_dotenv()


# --------------------------------------------------
# MODEL
# --------------------------------------------------

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)


# --------------------------------------------------
# SEARCH AGENT
# --------------------------------------------------

def build_search_agent():

    return create_agent(
        model=llm,
        tools=[web_search]
    )


# --------------------------------------------------
# READER AGENT
# --------------------------------------------------

def build_reader_agent():

    return create_agent(
        model=llm,
        tools=[scrape_url]
    )


# --------------------------------------------------
# WRITER CHAIN
# --------------------------------------------------

writer_prompt = ChatPromptTemplate.from_messages([

    (
        "system",
        "You are an expert research writer. "
        "Write clear, structured and insightful reports."
    ),

    (
        "human",
        """
        Write a detailed research report on the topic below.

        Topic:
        {topic}

        Research Gathered:
        {research}

        Structure the report as:

        1. Introduction
        2. Key Findings
        3. Economic / Technical Explanation
        4. Important Examples
        5. Conclusion
        6. Sources

        Rules:
        - Use ONLY the information given in Research Gathered. Do not add facts from your own knowledge.
        - Cite every important claim inline using numbers like [1], [2].
        - In the Sources section, list each number with its URL.
        - If some information is missing or unclear in the research, say so instead of guessing.

        Be factual, professional and well structured.
        """
    )
])


writer_chain = writer_prompt | llm | StrOutputParser()


# --------------------------------------------------
# CRITIC CHAIN
# --------------------------------------------------

critic_prompt = ChatPromptTemplate.from_messages([

    (
        "system",
        "You are a sharp and constructive research critic. "
        "Be honest and specific."
    ),

    (
        "human",
        """
        Review the research report below strictly.

        Report:
        {report}

        Respond ONLY in this format:

        Score: X/10

        Strengths:
        - ...
        - ...

        Areas to Improve:
        - ...
        - ...

        One line verdict:
        ...
        """
    )
])


critic_chain = critic_prompt | llm | StrOutputParser()