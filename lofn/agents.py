from crewai import Agent
from crewai_tools import SerperDevTool
import os

# Read environment variables
OPENAI_API = os.environ.get('OPENAI_API', '')
ANTHROPIC_API = os.environ.get('ANTHROPIC_API', '')

search_tool = SerperDevTool()

class EssenceAndFacetsAgent(Agent):
    def __init__(self):
        super().__init__(
            role='Essence and Facets Generator',
            goal='Generate essence and facets for the given input',
            tools=[search_tool],
            verbose=True,
            memory=True,
            backstory="You are an AI assistant specialized in generating the essence and facets of a concept."
        )

class ConceptsAgent(Agent):
    def __init__(self):
        super().__init__(
            role='Concepts Generator',
            goal='Generate concepts based on the provided essence and facets',
            tools=[search_tool],
            verbose=True,
            memory=True,
            backstory="You are an AI assistant specialized in generating concepts based on the provided essence and facets."
        )

class ArtistAndRefinedConceptsAgent(Agent):
    def __init__(self):
        super().__init__(
            role='Artist and Refined Concepts Generator',
            goal='Generate artists and refined concepts based on the provided essence, facets, and concepts',
            tools=[search_tool],
            verbose=True,
            memory=True,
            backstory="You are an AI assistant specialized in generating artists and refined concepts based on the provided essence, facets, and concepts."
        )

class MediumAgent(Agent):
    def __init__(self):
        super().__init__(
            role='Medium Generator',
            goal='Generate mediums based on the provided essence, facets, and refined concepts',
            tools=[search_tool],
            verbose=True,
            memory=True,
            backstory="You are an AI assistant specialized in generating mediums based on the provided essence, facets, and refined concepts."
        )

class RefineMediumAgent(Agent):
    def __init__(self):
        super().__init__(
            role='Refine Medium Generator',
            goal='Refine mediums based on the provided essence, facets, mediums, artists, and refined concepts',
            tools=[search_tool],
            verbose=True,
            memory=True,
            backstory="You are an AI assistant specialized in refining mediums based on the provided essence, facets, mediums, artists, and refined concepts."
        )

class ShuffledReviewAgent(Agent):
    def __init__(self):
        super().__init__(
            role='Shuffled Review Generator',
            goal='Generate shuffled reviews based on the provided essence, facets, mediums, artists, and refined concepts',
            tools=[search_tool],
            verbose=True,
            memory=True,
            backstory="You are an AI assistant specialized in generating shuffled reviews based on the provided essence, facets, mediums, artists, and refined concepts."
        )
