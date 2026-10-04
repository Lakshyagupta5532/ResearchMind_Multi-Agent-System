# ResearchMind: Multi-Agent Research System

A multi-agent research assistant. Give it a topic and four AI agents search the web, read the best sources, write a cited report and score it.

**Live demo:** https://researchmindmulti-agent-systems.streamlit.app/

## How it works

```
Topic -> Search Agent -> Reader Agent -> Writer Chain -> Critic Chain -> Report + Score
```

| Step | What it does |
|------|--------------|
| Search Agent | Uses the Tavily API to find recent web sources for the topic |
| Reader Agent | Picks the 3 most relevant URLs and scrapes each one with BeautifulSoup |
| Writer Chain | Writes a structured report using only the gathered research, with inline citations like [1], [2] |
| Critic Chain | Reviews the report and returns a score out of 10 with strengths and improvements |

## Features

- Live pipeline tracker showing the status and time of each agent
- Multi-source scraping (3 URLs per topic) for more depth
- Inline citations with a Sources section
- Automatic retry when the Groq API rate limit is hit
- Download the final report as Markdown

## Tech stack

Python, LangChain, Groq (gpt-oss-120b), Tavily Search API, BeautifulSoup, Streamlit

## Run locally

```bash
git clone https://github.com/Lakshyagupta5532/ResearchMind_Multi-Agent-System.git
cd ResearchMind_Multi-Agent-System
pip install -r requirements.txt
```

Create a `.env` file in the project folder:

```
GROQ_API_KEY=your_groq_key
TAVILY_API_KEY=your_tavily_key
```

Start the app:

```bash
streamlit run app.py
```

## Project structure

```
app.py            Streamlit UI and pipeline runner
agents.py         Search agent, reader agent, writer and critic chains
tools.py          web_search (Tavily) and scrape_url (BeautifulSoup) tools
pipeline.py       Command line version of the pipeline
requirements.txt  Python dependencies
.streamlit/       Streamlit theme config
```

## Limitations

- Runs on the Groq free tier, so heavy use can hit per-minute or per-day token limits
- Scraped text is truncated per page, so very long articles are only partly read
- The critic scores structure and clarity, it does not fact-check claims

## Credits

Based on a LangChain multi-agent tutorial. Extended with a custom Streamlit UI, multi-URL scraping, source citations and rate limit retry.
