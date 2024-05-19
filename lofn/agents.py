from crewai import Agent
from crewai_tools import SerperDevTool
import os

# Read environment variables
OPENAI_API = os.environ.get('OPENAI_API', '')
ANTHROPIC_API = os.environ.get('ANTHROPIC_API', '')

search_tool = SerperDevTool()

# Define CrewAI agents
essence_and_facets_agent = Agent(
    role='Essence and Facets Generator',
    goal='Generate essence and facets for the given input',
    tools=[search_tool],
    verbose=True,
    memory=True,
    backstory="You are an AI assistant specialized in generating the essence and facets of a concept."
)

concepts_agent = Agent(
    role='Concepts Generator',
    goal='Generate 12 concepts based on the provided essence and facets',
    tools=[search_tool],
    verbose=True,
    memory=True,
    backstory="You are an AI assistant specialized in generating concepts based on the provided essence and facets."
)

artist_and_refined_concepts_agent = Agent(
    role='Artist and Refined Concepts Generator',
    goal='Generate artists and refined concepts based on the provided essence, facets, and concepts',
    tools=[search_tool],
    verbose=True,
    memory=True,
    backstory="You are an AI assistant specialized in generating artists and refined concepts based on the provided essence, facets, and concepts."
)

medium_agent = Agent(
    role='Medium Generator',
    goal='Generate 12 mediums based on the provided essence, facets, and refined concepts',
    tools=[search_tool],
    verbose=True,
    memory=True,
    backstory="You are an AI assistant specialized in generating mediums based on the provided essence, facets, and refined concepts."
)

refine_medium_agent = Agent(
    role='Refine Medium Generator',
    goal='Refine mediums based on the provided essence, facets, mediums, artists, and refined concepts',
    tools=[search_tool],
    verbose=True,
    memory=True,
    backstory="You are an AI assistant specialized in refining mediums based on the provided essence, facets, mediums, artists, and refined concepts."
)

shuffled_review_agent = Agent(
    role='Shuffled Review Generator',
    goal='Generate shuffled reviews based on the provided essence, facets, mediums, artists, and refined concepts',
    tools=[search_tool],
    verbose=True,
    memory=True,
    backstory="You are an AI assistant specialized in generating shuffled reviews based on the provided essence, facets, mediums, artists, and refined concepts."
)
