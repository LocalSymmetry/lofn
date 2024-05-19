from crewai import Task
from lofn.agents import (
    essence_and_facets_agent,
    concepts_agent,
    artist_and_refined_concepts_agent,
    medium_agent,
    refine_medium_agent,
    shuffled_review_agent
)

# Define tasks
essence_and_facets_task = Task(
    description="Generate essence and facets for the given input.",
    expected_output="A JSON object with essence and facets.",
    tools=[search_tool],
    agent=essence_and_facets_agent
)

concepts_task = Task(
    description="Generate concepts based on the provided essence and facets.",
    expected_output="A JSON object with 12 concepts.",
    tools=[search_tool],
    agent=concepts_agent
)

artist_and_refined_concepts_task = Task(
    description="Generate artists and refined concepts based on the provided essence, facets, and concepts.",
    expected_output="A JSON object with 12 artists and 12 refined concepts.",
    tools=[search_tool],
    agent=artist_and_refined_concepts_agent
)

medium_task = Task(
    description="Generate mediums based on the provided essence, facets, and refined concepts.",
    expected_output="A JSON object with 12 mediums.",
    tools=[search_tool],
    agent=medium_agent
)

refine_medium_task = Task(
    description="Refine mediums based on the provided essence, facets, mediums, artists, and refined concepts.",
    expected_output="A JSON object with 12 refined mediums.",
    tools=[search_tool],
    agent=refine_medium_agent
)

shuffled_review_task = Task(
    description="Generate shuffled reviews based on the provided essence, facets, mediums, artists, and refined concepts.",
    expected_output="A JSON object with 12 shuffled reviews.",
    tools=[search_tool],
    agent=shuffled_review_agent
)
