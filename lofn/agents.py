from crewai import Agent
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
import random
import numpy as np
import os

# Read environment variables
OPENAI_API = os.environ.get('OPENAI_API', '')
ANTHROPIC_API = os.environ.get('ANTHROPIC_API', '')

def get_llm(model, temperature, openai_api_key=OPENAI_API, anthropic_api_key=ANTHROPIC_API):
    if model.startswith("claude"):
        return ChatAnthropic(model=model, temperature=temperature, max_tokens_to_sample=2048, anthropic_api_key=anthropic_api_key)
    else:
        return ChatOpenAI(model=model, temperature=temperature, openai_api_key=openai_api_key)

class EssenceAndFacetsAgent(Agent):
    def __init__(self, role, goal, tools=None):
        super().__init__(role=role, goal=goal, tools=tools)

    def execute(self, input, model, temperature, max_retries, concept_system, essence_prompt, debug=False):
        llm = get_llm(model, temperature)
        chain = LLMChain(
            llm=llm, 
            prompt=ChatPromptTemplate.from_messages([
                ("system", concept_system),
                ("human", essence_prompt)
            ])
        )
        return run_chain_with_retries(chain, {"input": input}, max_retries, debug)

class ConceptsAgent(Agent):
    def __init__(self, role, goal, tools=None):
        super().__init__(role=role, goal=goal, tools=tools)

    def execute(self, input, model, temperature, max_retries, concept_system, concepts_prompt, essence, facets, debug=False):
        llm = get_llm(model, temperature)
        chain = LLMChain(
            llm=llm, 
            prompt=ChatPromptTemplate.from_messages([
                ("system", concept_system),
                ("human", concepts_prompt)
            ])
        )
        return run_chain_with_retries(chain, {"input": input, "essence": essence, "facets": facets}, max_retries, debug)

class ArtistAndRefinedConceptsAgent(Agent):
    def __init__(self, role, goal, tools=None):
        super().__init__(role=role, goal=goal, tools=tools)

    def execute(self, input, model, temperature, max_retries, concept_system, artist_and_critique_prompt, essence, facets, concepts, debug=False):
        llm = get_llm(model, temperature)
        chain = LLMChain(
            llm=llm, 
            prompt=ChatPromptTemplate.from_messages([
                ("system", concept_system),
                ("human", artist_and_critique_prompt)
            ])
        )
        return run_chain_with_retries(chain, {"input": input, "essence": essence, "facets": facets, "concepts": concepts}, max_retries, debug)

class MediumAgent(Agent):
    def __init__(self, role, goal, tools=None):
        super().__init__(role=role, goal=goal, tools=tools)

    def execute(self, input, model, temperature, max_retries, concept_system, medium_prompt, essence, facets, refined_concepts, debug=False):
        llm = get_llm(model, temperature)
        chain = LLMChain(
            llm=llm, 
            prompt=ChatPromptTemplate.from_messages([
                ("system", concept_system),
                ("human", medium_prompt)
            ])
        )
        return run_chain_with_retries(chain, {"input": input, "essence": essence, "facets": facets, "refined_concepts": refined_concepts}, max_retries, debug)

class RefineMediumAgent(Agent):
    def __init__(self, role, goal, tools=None):
        super().__init__(role=role, goal=goal, tools=tools)

    def execute(self, input, model, temperature, max_retries, concept_system, refine_medium_prompt, essence, facets, mediums, artists, refined_concepts, debug=False):
        llm = get_llm(model, temperature)
        chain = LLMChain(
            llm=llm, 
            prompt=ChatPromptTemplate.from_messages([
                ("system", concept_system),
                ("human", refine_medium_prompt)
            ])
        )
        return run_chain_with_retries(chain, {"input": input, "essence": essence, "facets": facets, "mediums": mediums, "artists": artists, "refined_concepts": refined_concepts}, max_retries, debug)

class ShuffledReviewAgent(Agent):
    def __init__(self, role, goal, tools=None):
        super().__init__(role=role, goal=goal, tools=tools)

    def execute(self, input, model, temperature, max_retries, concept_system, refine_medium_prompt, essence, facets, mediums, artists, refined_concepts, debug=False):
        llm = get_llm(model, temperature)
        chain = LLMChain(
            llm=llm, 
            prompt=ChatPromptTemplate.from_messages([
                ("system", concept_system),
                ("human", refine_medium_prompt)
            ])
        )
        return run_chain_with_retries(chain, {"input": input, "essence": essence, "facets": facets, "mediums": mediums, "artists": artists, "refined_concepts": refined_concepts}, max_retries, debug)

def run_chain_with_retries(chain, args_dict, max_retries, debug=False):
    output = None
    retry_count = 0
    while retry_count < max_retries:
        try:
            output = chain.invoke(args_dict)
            break
        except Exception as e:
            if debug:
                print(f"An error occurred, retrying: {e}")
            retry_count += 1
    if retry_count >= max_retries:
        if debug:
            print("Max retries reached. Exiting.")
    return str(output)
