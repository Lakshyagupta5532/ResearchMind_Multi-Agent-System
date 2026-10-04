from agents import (
    build_search_agent,
    build_reader_agent,
    writer_chain,
    critic_chain
)


def run_research_pipeline(topic: str) -> dict:

    state = {}

    # --------------------------------------------------
    # STEP 1: SEARCH AGENT
    # --------------------------------------------------

    print("\n[bold green]Running Search Agent...[/bold green]")

    search_agent = build_search_agent()

    search_result = search_agent.invoke({
        "messages": [
            (
                "user",
                f"Find recent, reliable and detailed information about: {topic}. "
                "Return at least 5 relevant results with their URLs."
            )
        ]
    })

    state["search_results"] = search_result["messages"][-1].content

    print("[bold green]Search Agent Completed.[/bold green]")
    print(state["search_results"])


    # --------------------------------------------------
    # STEP 2: READER AGENT
    # --------------------------------------------------

    print("\n[bold green]Running Reader Agent...[/bold green]")

    reader_agent = build_reader_agent()

    reader_result = reader_agent.invoke({
        "messages": [
            (
                "user",
                f"""
                Topic: {topic}

                Here are the search results:

                {state["search_results"]}

                From these results, identify the most relevant URL.
                Then use the scrape_url tool to scrape that URL.
                Return the important information extracted from the page.
                Do NOT ask me for a URL.
                """
            )
        ]
    })

    state["scraped_content"] = reader_result["messages"][-1].content

    print("[bold green]Reader Agent Completed.[/bold green]")
    print(state["scraped_content"])


    # --------------------------------------------------
    # STEP 3: WRITER CHAIN
    # --------------------------------------------------

    print("\n[bold green]Running Writer Chain...[/bold green]")

    research_combined = (
        f"SEARCH RESULTS:\n"
        f"{state['search_results']}\n\n"
        f"SCRAPED CONTENT:\n"
        f"{state['scraped_content']}"
    )

    writer_result = writer_chain.invoke({
        "topic": topic,
        "research": research_combined
    })

    state["report"] = writer_result

    print("[bold green]Final Research Report Generated:[/bold green]")
    print(state["report"])


    # --------------------------------------------------
    # STEP 4: CRITIC CHAIN
    # --------------------------------------------------

    print("\n[bold green]Running Critic Chain...[/bold green]")

    critic_result = critic_chain.invoke({
        "report": state["report"]
    })

    state["feedback"] = critic_result

    print("[bold green]Critic Feedback:[/bold green]")
    print(state["feedback"])

    return state


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":

    topic = input("Enter the topic for research: ")

    run_research_pipeline(topic)