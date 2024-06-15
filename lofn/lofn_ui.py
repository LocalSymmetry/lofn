# lofn_ui.py

import streamlit as st
import os
import requests
from datetime import datetime
import json
import ast
from langchain.chains.structured_output.base import create_structured_output_runnable
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain_anthropic.experimental import ChatAnthropicTools
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.schema import OutputParserException
import numpy as np
import os
import pandas as pd
from string import Template
import json
import re
import random
import openai

st.set_page_config(page_title="Lofn - The AI Artist", page_icon=":art:", layout="wide")
# Read custom CSS file
with open("/lofn/style.css") as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Read environment variables
OPENAI_API = os.environ.get('OPENAI_API', '')
ANTHROPIC_API = os.environ.get('ANTHROPIC_API', '')
webhook_url = os.environ.get('WEBHOOK_URL', '')

# Ensure the OpenAI API key is set
openai.api_key = os.environ.get('OPENAI_API', '')


def save_image_locally(image_url, filename, directory = 'images'):
    try:
        response = requests.get(image_url)
        if response.status_code == 200:
            with open(f'/{directory}/{filename}', 'wb') as f:
                f.write(response.content)
            st.write(f"Image saved as /{directory}/{filename}")
        else:
            st.write(f"Failed to download image: {response.status_code}")
    except Exception as e:
        st.write(f"An error occurred while saving the image: {str(e)}")

def generate_image_dalle3(prompt):
    try:
        response = openai.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024"
        )
        image_url = response.data[0].url
        return image_url
    except Exception as e:
        st.write(f"An error occurred while generating the image with DALL-E 3: {str(e)}")
        return None

def read_prompt(file_path):
    with open(file_path, "r") as file:
        return file.read()

def parse_output(schema, output, debug = False, max_retries=3):
    retry_count = 0
    while retry_count < max_retries:
        try:
            response_schemas = [
                ResponseSchema(name=key, description=value["description"])
                for key, value in schema["properties"].items()
            ]

            if debug:
                st.write("Original output:")
                st.write(output)

            # Find the 'text' field within the output
            text_pattern = r"'text':\s*'.*?({.*}).*?'}"
            # 'text':\s*'.*?({.*?})'.*?"
            text_match = re.search(text_pattern, output, re.DOTALL)

            #st.write(text_match)

            if text_match:
                json_string = text_match.group(1)
                #st.write(json_string)

                # Replace escaped newline characters and backslashes
                json_string = json_string.replace('\\n', '').replace('\\\\', '\\')


                # Remove leading and trailing whitespace
                json_string = json_string.strip()

                # Replace single quotes with double quotes, except for escaped single quotes
                json_string = re.sub(r"(?<!\\)'", '"', json_string)

                # Remove backslashes before double quotes (used for escaping)
                json_string = json_string.replace('\\"', '"').replace("'","").replace("\\","'")


                if debug:
                    st.write("Extracted JSON string:")
                    st.write(json_string)

                parser = StructuredOutputParser.from_response_schemas(response_schemas)
                try:
                    # Parse the cleaned JSON string
                    parsed_output = json.loads(json_string)
                    return parser.parse(json.dumps(parsed_output))
                except (OutputParserException, json.JSONDecodeError) as e:
                    st.write(f"An error occurred while parsing the output trying fixes: {e}")
                    json_string = json_string + "}"
                    parsed_output = json.loads(json_string)
                    return parser.parse(json.dumps(parsed_output))
            else:
                st.write("No 'text' field found in the output.")
                return None
        except Exception as e:
            st.write(f"An error occurred while parsing the output, retrying: {e}")
            retry_count += 1
    st.write("Max retries reached. Parsing failed.")
    return None

def create_structured_output_runnable_wrapper(schema, llm, prompt):
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain

# Initialize session state
def initialize_session_state():
    if 'concept_mediums' not in st.session_state: 
        st.session_state['concept_mediums'] = None

    if 'pairs_to_try' not in st.session_state:
        st.session_state['pairs_to_try'] = [0]

    if 'webhook_url' not in st.session_state:
        st.session_state['webhook_url'] = webhook_url  

    if 'send_to_discord' not in st.session_state:
        st.session_state['send_to_discord'] = True

    if 'use_default_webhook' not in st.session_state:
        st.session_state['use_default_webhook'] = True 

    if 'concept_manual_mode' not in st.session_state:
        st.session_state['concept_manual_mode'] = False 

    # Initialize variables for manual mode data persistence
    for key in ['essence_and_facets_output', 'concepts_output', 'artist_and_refined_concepts_output', 'medium_output', 'refined_medium_output', 'shuffled_review_output']:
        if key not in st.session_state:
            st.session_state[key] = None

    for key in ['proceed_concepts_clicked', 'proceed_artist_refined_clicked', 'proceed_mediums_clicked', 'proceed_refined_mediums_clicked', 'proceed_shuffled_reviews_clicked', 'complete_all_steps_clicked']:
        if key not in st.session_state:
            st.session_state[key] = False

initialize_session_state()

st.title("LOFN - The AI Artist")

concept_system = read_prompt('/lofn/prompts/concept_system.txt')

prompt_system = read_prompt('/lofn/prompts/prompt_system.txt')

prompt_ending = read_prompt('/lofn/prompts/prompt_ending.txt')

concept_header_part1 = read_prompt('/lofn/prompts/concept_header.txt')

concept_header_part2 = read_prompt('/lofn/prompts/concept_header_pt2.txt')

prompt_header_part1 = read_prompt("/lofn/prompts/prompt_header.txt")

prompt_header_part2 = read_prompt("/lofn/prompts/prompt_header_pt2.txt")

essence_prompt_middle = read_prompt("/lofn/prompts/essence_prompt.txt")

concepts_prompt_middle = read_prompt("/lofn/prompts/concepts_prompt.txt")

artist_and_critique_prompt_middle = read_prompt("/lofn/prompts/artist_and_critique_prompt.txt")

medium_prompt_middle = read_prompt("/lofn/prompts/medium_prompt.txt")

refine_medium_prompt_middle = read_prompt("/lofn/prompts/refine_medium_prompt.txt")

facets_prompt_middle = read_prompt("/lofn/prompts/facets_prompt.txt")

aspects_traits_prompt_middle = read_prompt("/lofn/prompts/aspects_traits_prompts.txt")

midjourney_prompt_middle = read_prompt("/lofn/prompts/imagegen_prompt.txt")

artist_refined_prompt_middle = read_prompt("/lofn/prompts/artist_refined_prompt.txt")

revision_synthesis_prompt_middle = read_prompt("/lofn/prompts/revision_synthesis_prompt.txt")

dalle3_gen_prompt_middle = read_prompt("/lofn/prompts/dalle3_gen_prompt.txt")

dalle3_gen_prompt_nodiv_middle = read_prompt("/lofn/prompts/dalle3_gen_nodiv_prompt.txt")

image_title_prompt_middle = read_prompt("/lofn/prompts/image_title_prompt.txt")

# Read aesthetics from the file
with open('/lofn/prompts/aesthetics.txt', 'r') as file:
    aesthetics = file.read().split(', ')

# Randomly select 50 aesthetics
# concept_selected_aesthetics = random.sample(aesthetics, 100)
# concept_aesthetics_string = ', '.join(concept_selected_aesthetics)

# prompt_selected_aesthetics = random.sample(aesthetics, 100)
# prompt_aesthetics_string = ', '.join(prompt_selected_aesthetics)

prompt_header = prompt_header_part1 + prompt_header_part2

concept_header = concept_header_part1 + concept_header_part2

essence_prompt = concept_header + essence_prompt_middle + prompt_ending 

concepts_prompt = concept_header + concepts_prompt_middle + prompt_ending

artist_and_critique_prompt = concept_header + artist_and_critique_prompt_middle + prompt_ending

# Prompt template for choosing medium for each artist-refined concept
medium_prompt = concept_header + medium_prompt_middle + prompt_ending

# Prompt template for refining the medium choice based on artist's critique
refine_medium_prompt = concept_header + refine_medium_prompt_middle + prompt_ending

facets_prompt = concept_header + facets_prompt_middle + prompt_ending

# Defining the prompt template for List of Aspects and Traits
aspects_traits_prompt = prompt_header + aspects_traits_prompt_middle + prompt_ending

# Defining the prompt template for Midjourney Prompts
midjourney_prompt = prompt_header + midjourney_prompt_middle + prompt_ending


# Defining the prompt template for Artist Selection and Refined Prompts
artist_refined_prompt = prompt_header + artist_refined_prompt_middle + prompt_ending

revision_synthesis_prompt = concept_header + revision_synthesis_prompt_middle + prompt_ending

dalle3_gen_prompt = dalle3_gen_prompt_middle + prompt_ending

dalle3_gen_nodiv_prompt = dalle3_gen_prompt_nodiv_middle + prompt_ending

image_title_prompt = prompt_header + image_title_prompt_middle + prompt_ending 

# Extended JSON Schema for Essence and Facets
essence_and_facets_schema = {
    "type": "object",
    "properties": {
        "essence_and_facets": {
            "type": "object",
            "properties": {
                "essence": {"type": "string", "description": "The essence of the concept"},
                "facets": {
                    "type": "array",
                    "items": {"type": "string", "description": "A facet of the concept"},
                    "minItems": 5,
                    "maxItems": 5,
                    "description": "The facets of the concept"
                }
            },
            "required": ["essence", "facets"],
            "description": "The essence and facets of the concept"
        }
    },
    "required": ["essence_and_facets"]
}


# Modified JSON Schema for Concepts
concepts_schema = {
    "type": "object",
    "properties": {
        "concepts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "concept": {"type": "string", "description": "A generated concept"}
                },
                "required": ["concept"],
                "description": "A generated concept"
            },
            "minItems": 10,
            "maxItems": 10,
            "description": "The generated concepts"
        }
    },
    "required": ["concepts"]
}

artist_and_refined_concepts_schema = {
    "type": "object",
    "properties": {
        "artists": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "artist": {"type": "string", "description": "An artist"}
                },
                "required": ["artist"],
                "description": "An artist"
            },
            "minItems": 10,
            "maxItems": 10,
            "description": "The selected artists"
        },
        "refined_concepts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "refined_concept": {"type": "string", "description": "A refined concept"}
                },
                "required": ["refined_concept"],
                "description": "A refined concept"
            },
            "minItems": 10,
            "maxItems": 10,
            "description": "The refined concepts"
        }
    },
    "required": ["artists", "refined_concepts"]
}


# JSON Schema for Medium
medium_schema = {
    "type": "object",
    "properties": {
        "mediums": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "medium": {"type": "string", "description": "A medium"}
                },
                "required": ["medium"],
                "description": "A medium"
            },
            "minItems": 10,
            "maxItems": 10,
            "description": "The selected mediums"
        }
    },
    "required": ["mediums"]
}

refined_medium_schema = {
    "type": "object",
    "properties": {
        "refined_mediums": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "refined_medium": {"type": "string", "description": "A refined medium"}
                },
                "required": ["refined_medium"],
                "description": "A refined medium"
            },
            "minItems": 10,
            "maxItems": 10,
            "description": "The refined mediums"
        }
    },
    "required": ["refined_mediums"]
}


# JSON Schema for Shuffled Review
shuffled_review_schema = {
    "type": "object",
    "properties": {
        "refined_mediums": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "refined_medium": {"type": "string", "description": "A refined medium"}
                },
                "required": ["refined_medium"],
                "description": "A refined medium"
            },
            "minItems": 10,
            "maxItems": 10,
            "description": "The shuffled refined mediums"
        }
    },
    "required": ["refined_mediums"]
}

facets_schema = {
    "type": "object",
    "properties": {
        "facets": {
            "type": "array",
            "items": {"type": "string", "description": "A facet of the concept"},
            "description": "The facets of the concept"
        }
    },
    "required": ["facets"]
}

aspects_traits_schema = {
    "type": "object",
    "properties": {
        "artistic_guides": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "artistic_guide": {"type": "string", "description": "An artistic guide"}
                },
                "required": ["artistic_guide"],
                "description": "An artistic guide"
            },
            "minItems": 6,
            "maxItems": 6,
            "description": "The artistic guides"
        }
    },
    "required": ["artistic_guides"]
}

midjourney_schema = {
    "type": "object",
    "properties": {
        "image_gen_prompts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "image_gen_prompt": {"type": "string", "description": "An image generator prompt"}
                },
                "required": ["image_gen_prompt"],
                "description": "A image generator prompt"
            },
            "minItems": 6,
            "maxItems": 6,
            "description": "The image generator prompts"
        }
    },
    "required": ["image_gen_prompts"]
}

artist_refined_schema = {
    "type": "object",
    "properties": {
        "artist_refined_prompts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "artist_refined_prompt": {"type": "string", "description": "An artist-refined prompt"}
                },
                "required": ["artist_refined_prompt"],
                "description": "An artist-refined prompt"
            },
            "minItems": 6,
            "maxItems": 6,
            "description": "The artist-refined prompts"
        }
    },
    "required": ["artist_refined_prompts"]
}

revision_synthesis_schema = {
    "type": "object",
    "properties": {
        "revised_prompts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "revised_prompt": {"type": "string", "description": "A revised prompt"}
                },
                "required": ["revised_prompt"],
                "description": "A revised prompt"
            },
            "description": "The revised prompts"
        },
        "synthesized_prompts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "synthesized_prompt": {"type": "string", "description": "A synthesized prompt"}
                },
                "required": ["synthesized_prompt"],
                "description": "A synthesized prompt"
            },
            "description": "The synthesized prompts"
        }
    },
    "required": ["revised_prompts", "synthesized_prompts"]
}

def get_llm(model, temperature, openai_api_key=OPENAI_API, anthropic_api_key=ANTHROPIC_API):
    if model.startswith("claude"):
        return ChatAnthropic(model=model, temperature=temperature, max_tokens=4096, anthropic_api_key=anthropic_api_key)
    else:
        return ChatOpenAI(model=model, temperature=temperature, max_tokens=4096, openai_api_key=openai_api_key)

def run_chain_with_retries(_lang_chain, max_retries, args_dict=None):
    output = None
    retry_count = 0
    while retry_count < max_retries:
        try:
            output = _lang_chain.invoke(args_dict)
            break
        except Exception as e:
            st.write(f"An error occurred, retrying: {e}")
            retry_count += 1
    if retry_count >= max_retries:
        st.write("Max retries reached. Exiting.")
    return str(output)

@st.cache_data(persist=True, experimental_allow_widgets=True)
def generate_image_title(input, concept, medium, image, image_prompt, max_retries, temperature, model, verbose=False, debug=False):
    llm = get_llm(model, temperature, OPENAI_API, ANTHROPIC_API)

    chain = LLMChain(llm=llm, prompt=ChatPromptTemplate.from_messages([("human", image_title_prompt)]))
    
    output = run_chain_with_retries(chain, args_dict={
        "input": input,
        "concept": concept,
        "medium": medium,
        "facets": st.session_state.facets_output['facets'],
        "image": image,
        "image_prompt": image_prompt
    }, max_retries=max_retries)

    title_schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "The final title"}
        },
        "required": ["title"]
    }

    title_output = parse_output(title_schema, output, debug, max_retries=max_retries)

    return title_output["title"]

def send_prompts_to_discord(prompts, premessage='Prompts:'):
    try:
        if st.session_state['send_to_discord']:
            requests.post(st.session_state['webhook_url'], data=json.dumps({"content": 'INCOMING MESSAGE: ' + premessage}), headers={"Content-Type": "application/json"})
        for prompt in prompts:
            message_to_discord = {"content": prompt}
            if st.session_state['send_to_discord']:
                requests.post(st.session_state['webhook_url'], data=json.dumps(message_to_discord), headers={"Content-Type": "application/json"})
    except Exception as e:
        st.write(f"An error occurred while sending concepts to Discord: {str(e)}")

def send_concepts_to_discord(input, concept_list, premessage='Concepts and:'):
    try:
        if st.session_state['send_to_discord']:
            requests.post(st.session_state['webhook_url'], data=json.dumps({"content": f'INCOMING MESSAGE for Patron\'s idea: {input}'}), headers={"Content-Type": "application/json"})

        for con_dict in concept_list:
            message_to_discord = {"content": '{concept} = ' + con_dict['concept'] + ' \n' + '{medium} = ' + con_dict['medium']}
            if st.session_state['send_to_discord']:
                requests.post(st.session_state['webhook_url'], data=json.dumps(message_to_discord), headers={"Content-Type": "application/json"})
    except Exception as e:
        st.write(f"An error occurred while sending concepts to Discord: {str(e)}")

def validate_facets(facets):
    if len(facets) >= 5:
        st.session_state.essence_and_facets_output = {
            "essence_and_facets": {
                "essence": essence,
                "facets": facets[:5]
            }
        }
    else:
        st.warning("Please provide at least 5 facets.")

def validate_concepts(concepts):
    if len(concepts) >= 10:
        st.session_state.concepts_output = {
            "concepts": [{"concept": concept} for concept in concepts[:10]]
        }
    else:
        st.warning("Please provide at least 10 concepts.")

def validate_artists_and_refined_concepts(artists, refined_concepts):
    if len(artists) >= 10 and len(refined_concepts) >= 10:
        st.session_state.artist_and_refined_concepts_output = {
            "artists": [{"artist": artist} for artist in artists[:10]],
            "refined_concepts": [{"refined_concept": rc} for rc in refined_concepts[:10]]
        }
    else:
        st.warning("Please provide at least 10 artists and 10 refined concepts.")

def validate_mediums(mediums):
    if len(mediums) >= 10:
        st.session_state.medium_output = {
            "mediums": [{"medium": medium} for medium in mediums[:10]]
        }
    else:
        st.warning("Please provide at least 10 mediums.")

def validate_refined_mediums(refined_mediums):
    if len(refined_mediums) >= 10:
        st.session_state.refined_medium_output = {
            "refined_mediums": [{"refined_medium": rm} for rm in refined_mediums[:10]]
        }
    else:
        st.warning("Please provide at least 10 refined mediums.")

def validate_shuffled_refined_mediums(shuffled_refined_mediums):
    if len(shuffled_refined_mediums) >= 10:
        st.session_state.shuffled_review_output = {
            "refined_mediums": [{"refined_medium": srm} for srm in shuffled_refined_mediums[:10]]
        }
    else:
        st.warning("Please provide at least 10 refined mediums for shuffled reviews.")

def generate_concept_mediums_manual(input, max_retries, temperature, model = "gpt-3.5-turbo-16k", verbose = False, debug = False):
 # Initialize variables to store output for each step
    if 'essence_and_facets_output' not in st.session_state:
        st.session_state.essence_and_facets_output = None
    if 'concepts_output' not in st.session_state:
        st.session_state.concepts_output = None
    if 'artist_and_refined_concepts_output' not in st.session_state:
        st.session_state.artist_and_refined_concepts_output = None
    if 'medium_output' not in st.session_state:
        st.session_state.medium_output = None
    if 'refined_medium_output' not in st.session_state:
        st.session_state.refined_medium_output = None
    if 'shuffled_review_output' not in st.session_state:
        st.session_state.shuffled_review_output = None

    if 'proceed_concepts_clicked' not in st.session_state:
        st.session_state.proceed_concepts_clicked = False
    if 'proceed_artist_refined_clicked' not in st.session_state:
        st.session_state.proceed_artist_refined_clicked = False
    if 'proceed_mediums_clicked' not in st.session_state:
        st.session_state.proceed_mediums_clicked = False
    if 'proceed_refined_mediums_clicked' not in st.session_state:
        st.session_state.proceed_refined_mediums_clicked = False
    if 'proceed_shuffled_reviews_clicked' not in st.session_state:
        st.session_state.proceed_shuffled_reviews_clicked = False
    if 'complete_all_steps_clicked' not in st.session_state:
        st.session_state.complete_all_steps_clicked = False

    # Initialize flag for step completion
    all_steps_completed = False

 # Concepts Manual mode
    st.write("Please provide the following details manually:")
    st.write("System Message")
    st.code(concept_system)

    st.code(essence_prompt.format(input=input))
    essence = st.text_input("Essence:")
    facets_str = st.text_area("Facets (enter separated by a newline or comma):")
    
    # Convert the facets to a list
    facets = [facet.strip() for facet in facets_str.split("\n" if "\n" in facets_str else ",")]

    # Validate facets
    validate_facets(facets)

    if st.button("Proceed to Concepts"):
        st.session_state.proceed_concepts_clicked = True

    if st.session_state.proceed_concepts_clicked:
        if st.session_state.essence_and_facets_output:
            # Manual mode for concepts_output
            st.write("Please provide the following concepts manually:")
            st.code(concepts_prompt.format(essence=essence, facets=str(facets), input=input))
            concepts_str = st.text_area("Concepts (separated by a newline or comma):")

            # Convert the concepts to a list
            concepts = [concept.strip() for concept in concepts_str.split("\n" if "\n" in concepts_str else ",")]

            # Validate concepts
            validate_concepts(concepts)
        else:
            st.warning("Please complete all previous steps before proceeding.")

    if st.button("Proceed to Artists and Refined Concepts"):
        st.session_state.proceed_artist_refined_clicked = True
    
    if st.session_state.proceed_artist_refined_clicked: 
        if st.session_state.essence_and_facets_output and st.session_state.concepts_output:
            
            st.write("Please provide the following artists and refined concepts manually:")
            st.code(artist_and_critique_prompt.format(essence=essence, facets=str(facets), concepts=str(concepts), input=input))
            artists_str = st.text_area("Artists (separated by a newline or comma):")
            refined_concepts_str = st.text_area("Refined Concepts (separated by a newline or comma):")

            # Convert the artists and refined concepts to lists
            artists = [artist.strip() for artist in artists_str.split("\n" if "\n" in artists_str else ",")]
            refined_concepts = [rc.strip() for rc in refined_concepts_str.split("\n" if "\n" in refined_concepts_str else ",")]

            # Validate artists and refined concepts
            validate_artists_and_refined_concepts(artists, refined_concepts)
        else:
            st.warning("Please complete all previous steps before proceeding.")

    if st.button("Proceed to Mediums"):
        st.session_state.proceed_mediums_clicked = True
    
    if st.session_state.proceed_mediums_clicked:
        if st.session_state.essence_and_facets_output and st.session_state.concepts_output and st.session_state.artist_and_refined_concepts_output:
            st.write("Please provide the following mediums manually:")
            st.code(medium_prompt.format(refined_concepts=str(refined_concepts), input=input))
            mediums_str = st.text_area("Mediums (separated by a newline or comma):")
            mediums = [medium.strip() for medium in mediums_str.split("\n" if "\n" in mediums_str else ",")]

            # Validate mediums
            validate_mediums(mediums)
        else:
            st.warning("Please complete all previous steps before proceeding.")
    
    if st.button("Proceed to Refined Mediums"):
        st.session_state.proceed_refined_mediums_clicked = True

    if st.session_state.proceed_refined_mediums_clicked:
        if st.session_state.essence_and_facets_output and st.session_state.concepts_output and st.session_state.artist_and_refined_concepts_output and st.session_state.medium_output:
            st.write("Please provide the following refined mediums manually:")
            st.code(refine_medium_prompt.format(essence=essence, facets=str(facets), mediums=str(mediums), artists=str(artists), refined_concepts=str(refined_concepts), input=input))
            refined_mediums_str = st.text_area("Refined Mediums (separated by a newline or comma):")
            refined_mediums = [rm.strip() for rm in refined_mediums_str.split("\n" if "\n" in refined_mediums_str else ",")]

            # Validate refined mediums
            validate_refined_mediums(refined_mediums)
        else:
            st.warning("Please complete all previous steps before proceeding.")

    if st.button("Proceed to Shuffled Reviews"):
        st.session_state.proceed_shuffled_reviews_clicked = True

    if st.session_state.proceed_shuffled_reviews_clicked:
        if st.session_state.essence_and_facets_output and st.session_state.concepts_output and st.session_state.artist_and_refined_concepts_output and st.session_state.medium_output and st.session_state.refined_medium_output:    
            st.write("Please provide the following refined mediums for shuffled reviews manually:")
            st.code(refine_medium_prompt.format(essence=essence, facets=str(facets), mediums=str(mediums), artists=str(np.random.permutation(artists)), refined_concepts=str(refined_concepts), input=input))
            shuffled_refined_mediums_str = st.text_area("Shuffled Refined Mediums (separated by a newline or comma):")
            shuffled_refined_mediums = [srm.strip() for srm in shuffled_refined_mediums_str.split("\n" if "\n" in shuffled_refined_mediums_str else ",")]

            # Validate shuffled refined mediums
            validate_shuffled_refined_mediums(shuffled_refined_mediums)
            
        else:
            st.warning("Please complete all previous steps before proceeding.")
                        
    # Add a condition to set the all_steps_completed flag
    if st.button("Complete All Concept and Medium Steps"):
        st.session_state.complete_all_steps_clicked = True

    if st.session_state.complete_all_steps_clicked:
        if st.session_state.essence_and_facets_output and st.session_state.concepts_output and st.session_state.artist_and_refined_concepts_output and st.session_state.medium_output and st.session_state.refined_medium_output and st.session_state.shuffled_review_output:
            all_steps_completed = True
        else:
            st.warning("Please complete all steps before proceeding.")
                    
    if all_steps_completed:
        refined_concepts = [x['refined_concept'] for x in st.session_state.artist_and_refined_concepts_output['refined_concepts']]
        refined_mediums = [x['refined_medium'] for x in st.session_state.shuffled_review_output['refined_mediums']]
        concept_mediums = [{'concept': concept, 'medium': medium} for concept, medium in zip(refined_concepts, refined_mediums)]
        
        send_concepts_to_discord(input, concept_mediums)
        return concept_mediums

@st.cache_data(persist=True, experimental_allow_widgets=True)
def generate_concept_mediums(input, max_retries, temperature, model="gpt-3.5-turbo-16k", verbose=False, debug=False, aesthetics=aesthetics):
    # Initialize session state variables
    for key in [
        'essence_and_facets_output', 'concepts_output', 'artist_and_refined_concepts_output',
        'medium_output', 'refined_medium_output', 'shuffled_review_output'
    ]:
        if key not in st.session_state:
            st.session_state[key] = None

    llm = get_llm(model, temperature, OPENAI_API, ANTHROPIC_API)

    selected_aesthetics = random.sample(aesthetics, 100)
    aesthetics_string = ', '.join(selected_aesthetics)

    chains = {
        'essence_and_facets': LLMChain(
            llm=llm, 
            prompt=ChatPromptTemplate.from_messages([
                ("system", concept_system),
                ("human", essence_prompt)
            ])
        ),
        'concepts': LLMChain(
            llm=llm, 
            prompt=ChatPromptTemplate.from_messages([
                ("system", concept_system),
                ("human", concepts_prompt)
            ])
        ),
        'artist_and_refined_concepts': LLMChain(
            llm=llm, 
            prompt=ChatPromptTemplate.from_messages([
                ("system", concept_system),
                ("human", artist_and_critique_prompt)
            ])
        ),
        'medium': LLMChain(
            llm=llm, 
            prompt=ChatPromptTemplate.from_messages([
                ("system", concept_system),
                ("human", medium_prompt)
            ])
        ),
        'refine_medium': LLMChain(
            llm=llm, 
            prompt=ChatPromptTemplate.from_messages([
                ("system", concept_system),
                ("human", refine_medium_prompt)
            ])
        ),
        'shuffled_review': LLMChain(
            llm=llm, 
            prompt=ChatPromptTemplate.from_messages([
                ("system", concept_system),
                ("human", refine_medium_prompt)
            ])
        )
    }

    def run_chain(chain_name, args_dict, aesthetics=selected_aesthetics):
        args_dict["randomaesthetics"] = aesthetics
        return run_chain_with_retries(chains[chain_name], args_dict=args_dict, max_retries=max_retries)

    st.session_state.essence_and_facets_output = parse_output(essence_and_facets_schema, run_chain('essence_and_facets', {"input": input}), debug, max_retries=max_retries)
    if debug:
        st.write(f"essence_and_facets_output: {st.session_state.essence_and_facets_output}")

    st.session_state.concepts_output = parse_output(concepts_schema, run_chain('concepts', {
        "input": input,
        "essence": st.session_state.essence_and_facets_output["essence_and_facets"]["essence"],
        "facets": st.session_state.essence_and_facets_output["essence_and_facets"]["facets"]
    }), 
    debug, 
    max_retries=max_retries)
    if debug:
        st.write(f"concepts_output: {st.session_state.concepts_output}")

    st.session_state.artist_and_refined_concepts_output = parse_output(artist_and_refined_concepts_schema, run_chain('artist_and_refined_concepts', {
        "input": input,
        "essence": st.session_state.essence_and_facets_output["essence_and_facets"]["essence"],
        "facets": st.session_state.essence_and_facets_output["essence_and_facets"]["facets"],
        "concepts": [x['concept'] for x in st.session_state.concepts_output['concepts']]
    }), 
    debug, 
    max_retries=max_retries)
    if debug:
        st.write(f"artist_and_critique_list: {st.session_state.artist_and_refined_concepts_output}")

    st.session_state.medium_output = parse_output(medium_schema, run_chain('medium', {
        "input": input,
        "essence": st.session_state.essence_and_facets_output["essence_and_facets"]["essence"],
        "facets": st.session_state.essence_and_facets_output["essence_and_facets"]["facets"],
        "refined_concepts": [x['refined_concept'] for x in st.session_state.artist_and_refined_concepts_output['refined_concepts']]
    }), 
    debug, 
    max_retries=max_retries)
    if debug:
        st.write(f"medium_list: {st.session_state.medium_output}")

    st.session_state.refined_medium_output = parse_output(refined_medium_schema, run_chain('refine_medium', {
        "input": input,
        "essence": st.session_state.essence_and_facets_output["essence_and_facets"]["essence"],
        "facets": st.session_state.essence_and_facets_output["essence_and_facets"]["facets"],
        "mediums": [x['medium'] for x in st.session_state.medium_output['mediums']],
        "artists": [x['artist'] for x in st.session_state.artist_and_refined_concepts_output['artists']],
        "refined_concepts": [x['refined_concept'] for x in st.session_state.artist_and_refined_concepts_output['refined_concepts']]
    }), 
    debug, 
    max_retries=max_retries)
    if debug:
        st.write(f"refined_medium_list: {st.session_state.refined_medium_output}")

    review_artists = np.random.permutation([x['artist'] for x in st.session_state.artist_and_refined_concepts_output['artists']]).tolist()

    st.session_state.shuffled_review_output = parse_output(shuffled_review_schema, run_chain('shuffled_review', {
        "input": input,
        "essence": st.session_state.essence_and_facets_output["essence_and_facets"]["essence"],
        "facets": st.session_state.essence_and_facets_output["essence_and_facets"]["facets"],
        "mediums": [x['medium'] for x in st.session_state.medium_output['mediums']],
        "artists": review_artists,
        "refined_concepts": [x['refined_concept'] for x in st.session_state.artist_and_refined_concepts_output['refined_concepts']]
    }), 
    debug, 
    max_retries=max_retries)
    if debug:
        st.write(f"shuffled_review_list: {st.session_state.shuffled_review_output}")

    refined_concepts = [x['refined_concept'] for x in st.session_state.artist_and_refined_concepts_output['refined_concepts']]
    refined_mediums = [x['refined_medium'] for x in st.session_state.shuffled_review_output['refined_mediums']]
    concept_mediums = [{'concept': concept, 'medium': medium} for concept, medium in zip(refined_concepts, refined_mediums)]
    send_concepts_to_discord(input, concept_mediums)
    return concept_mediums
    
def generate_prompts_manual(input, concept, medium, max_retries, temperature, model="gpt-3.5-turbo-16k", verbose=False, debug=False):
    # Initialize session state variables if they do not exist
    for key in ['facets_output', 'artistic_guides_output', 'midjourney_output', 'artist_refined_output', 'revised_synthesized_prompts_output']:
        if key not in st.session_state:
            st.session_state[key] = None

    for key in ['proceed_facets', 'proceed_artistic_guides', 'proceed_midjourney', 'proceed_artist_refined_prompts', 'proceed_revision_synthesis', 'complete_all_prompts_steps']:
        if key not in st.session_state:
            st.session_state[key] = False

    st.write("Manual mode for generating prompts is activated.")

    # Facets
    st.write("Please provide facets manually:")
    st.code(facets_prompt.format(input=input, concept=concept, medium=medium))
    facets_prompt_str = st.text_area("Facets (separated by a newline or comma):")
    facets = [facet.strip() for facet in facets_prompt_str.split("\n" if "\n" in facets_prompt_str else ",")]

    if len(facets) >= 1:
        st.session_state.facets_output = {
            "facets": facets
        }
        if st.button("Proceed to Artistic Guides"):  # Moved outside of the above block
            st.session_state.proceed_facets = True
    else:
        st.warning("Please provide at least 5 facets.")

    # Artistic Guides
    if st.session_state.proceed_facets:
        st.write("Please provide artistic guides manually:")
        # Assume aspects_traits_prompt is defined elsewhere in your code
        st.code(aspects_traits_prompt.format(input=input, concept=concept, medium=medium, facets=facets))
        artistic_guides_str = st.text_area("Artistic Guides (separated by a newline or comma):")
        artistic_guides = [guide.strip() for guide in artistic_guides_str.split("\n" if "\n" in artistic_guides_str else ",")]

        # Validate based on the schema
        if len(artistic_guides) >= 6:
            st.session_state.artistic_guides_output = {
                "artistic_guides": [{"artistic_guide": guide} for guide in artistic_guides[:6]]
            }
            if st.button("Proceed to generating Midjourney Prompts"):
                st.session_state.proceed_artistic_guides = True
        else:
            st.warning("Please provide at least 6 artistic guides.")

    # Midjourney Prompts
    if st.session_state.proceed_artistic_guides:
        st.write("Please provide the following Midjourney prompts manually:")
        # Assume midjourney_prompt is defined elsewhere in your code
        st.code(midjourney_prompt.format(concept=concept, medium=medium, facets=facets, artistic_guides=artistic_guides, input=input))
        midjourney_prompts_str = st.text_area("Midjourney Prompts (separated by a newline or comma):")
        midjourney_prompts = [prompt.strip() for prompt in midjourney_prompts_str.split("\n" if "\n" in midjourney_prompts_str else ",")]

        # Validate based on the schema
        if len(midjourney_prompts) == 6:
            st.session_state.midjourney_output = {
                "image_gen_prompts": [{"image_gen_prompt": prompt} for prompt in midjourney_prompts]
            }
        if st.button("Proceed to Artist Refined Prompts"):
            st.session_state.proceed_midjourney = True
        else:
            st.warning("Please provide exactly 6 Midjourney prompts.")

    # Artist Refined Prompts
    if st.session_state.proceed_midjourney:
        st.write("Please provide artist-refined prompts manually:")
        
        # Assume artist_refined_prompt is defined elsewhere in your code
        st.code(artist_refined_prompt.format(
            concept=concept, 
            medium=medium, 
            facets=st.session_state.facets_output['facets'], 
            input=input,
            image_gen_prompts=[x['image_gen_prompt'] for x in st.session_state.midjourney_output['image_gen_prompts']]
        ))
        
        artist_refined_prompts_str = st.text_area("Artist Refined Prompts (separated by a newline or comma):")
        artist_refined_prompts = [prompt.strip() for prompt in artist_refined_prompts_str.split("\n" if "\n" in artist_refined_prompts_str else ",")]
        
        # Validate based on the schema
        if len(artist_refined_prompts) == 6:
            st.session_state.artist_refined_output = {
                "artist_refined_prompts": [{"artist_refined_prompt": prompt} for prompt in artist_refined_prompts]
            }
            if st.button("Proceed to Revision and Synthesis"):
                st.session_state.proceed_artist_refined_prompts = True
        else:
            st.warning("Please provide exactly 6 artist refined prompts.")

    # Revision and Synthesis
    if st.session_state.proceed_artist_refined_prompts:
        st.write("Please provide revised and synthesized prompts manually:")
        
        # Assume revision_synthesis_prompt is defined elsewhere in your code
        st.code(revision_synthesis_prompt.format(
            concept=concept, 
            medium=medium, 
            facets=st.session_state.facets_output['facets'], 
            input=input,
            artist_refined_prompts=[x['artist_refined_prompt'] for x in st.session_state.artist_refined_output['artist_refined_prompts']]
        ))

        revised_prompts_str = st.text_area("Revised Prompts (separated by a newline or comma):")
        revised_prompts = [prompt.strip() for prompt in revised_prompts_str.split("\n" if "\n" in revised_prompts_str else ",")]

        synthesized_prompts_str = st.text_area("Synthesized Prompts (separated by a newline or comma):")
        synthesized_prompts = [prompt.strip() for prompt in synthesized_prompts_str.split("\n" if "\n" in synthesized_prompts_str else ",")]

        if len(revised_prompts) == 6 and len(synthesized_prompts) == 6:
            st.session_state.revised_synthesized_prompts_output = {
                "revised_prompts": [{"revised_prompt": prompt} for prompt in revised_prompts],
                "synthesized_prompts": [{"synthesized_prompt": prompt} for prompt in synthesized_prompts]
            }
            if st.button("Complete All Prompt Steps"):
                st.session_state.proceed_revision_synthesis = True
        else:
            st.warning("Please provide exactly 6 revised and 6 synthesized prompts.")

    # Complete All Prompt Steps
    if st.session_state.proceed_revision_synthesis:
        if st.button("Finalize and Return Results"):
            st.session_state.complete_all_prompts_steps = True

    # Returning the Dataframe
    if st.session_state.complete_all_prompts_steps:
        df_prompts = pd.DataFrame({
            'Revised Prompts': [prompt['revised_prompt'] for prompt in st.session_state.revised_synthesized_prompts_output['revised_prompts']],
            'Synthesized Prompts': [prompt['synthesized_prompt'] for prompt in st.session_state.revised_synthesized_prompts_output['synthesized_prompts']]
        })
        return df_prompts

     

@st.cache_data(persist=True, experimental_allow_widgets=True)
def generate_prompts(input, concept, medium, max_retries, temperature, model="gpt-3.5-turbo-16k", verbose=False, debug=False, aesthetics=aesthetics):
    # Initialize session state variables
    for key in [
        'facets_output', 'artistic_guides_output', 'midjourney_prompts_output',
        'artist_refined_prompts_output', 'revised_synthesized_prompts_output'
    ]:
        if key not in st.session_state:
            st.session_state[key] = None

    llm = get_llm(model, temperature, OPENAI_API, ANTHROPIC_API)

    selected_aesthetics = random.sample(aesthetics, 100)
    aesthetics_string = ', '.join(selected_aesthetics)

    chains = {
        'facets': LLMChain(
            llm=llm, 
            prompt=ChatPromptTemplate.from_messages([
                ("system", concept_system),
                ("human", facets_prompt)
            ])
        ),
        'aspects_traits': LLMChain(
            llm=llm, 
            prompt=ChatPromptTemplate.from_messages([
                ("system", prompt_system),
                ("human", aspects_traits_prompt)
            ])
        ),
        'midjourney': LLMChain(
            llm=llm, 
            prompt=ChatPromptTemplate.from_messages([
                ("system", prompt_system),
                ("human", midjourney_prompt)
            ])
        ),
        'artist_refined': LLMChain(
            llm=llm, 
            prompt=ChatPromptTemplate.from_messages([
                ("system", prompt_system),
                ("human", artist_refined_prompt)
            ])
        ),
        'revision_synthesis': LLMChain(
            llm=llm, 
            prompt=ChatPromptTemplate.from_messages([
                ("system", prompt_system),
                ("human", revision_synthesis_prompt)
            ])
        )
    }

    def run_chain(chain_name, args_dict, aesthetics=selected_aesthetics):
        args_dict["randomaesthetics"] = aesthetics
        return run_chain_with_retries(chains[chain_name], args_dict=args_dict, max_retries=max_retries)

    st.session_state.facets_output = parse_output(facets_schema, run_chain('facets', {"input": input, "concept": concept, "medium": medium}), debug, max_retries=max_retries)
    if debug:
        st.write(f"facets_output: {st.session_state.facets_output}")

    st.session_state.artistic_guides_output = parse_output(aspects_traits_schema, run_chain('aspects_traits', {
        "input": input,
        "concept": concept,
        "medium": medium,
        "facets": st.session_state.facets_output['facets']
    }), 
    debug, 
    max_retries=max_retries)
    if debug:
        st.write(f"aspects_output: {st.session_state.artistic_guides_output}")

    st.session_state.midjourney_prompts_output = parse_output(midjourney_schema, run_chain('midjourney', {
        "input": input,
        "concept": concept,
        "medium": medium,
        "facets": st.session_state.facets_output['facets'],
        "artistic_guides": [x['artistic_guide'] for x in st.session_state.artistic_guides_output['artistic_guides']]
    }), 
    debug, 
    max_retries=max_retries)
    send_prompts_to_discord(
        [prompt['image_gen_prompt'] for prompt in st.session_state.midjourney_prompts_output['image_gen_prompts']],
        premessage=f'Generated Prompts for {concept} in {medium}:'
    )
    if debug:
        st.write(f"midjourney_output: {st.session_state.midjourney_prompts_output}")

    st.session_state.artist_refined_prompts_output = parse_output(artist_refined_schema, run_chain('artist_refined', {
        "input": input,
        "concept": concept,
        "medium": medium,
        "facets": st.session_state.facets_output['facets'],
        "image_gen_prompts": [x['image_gen_prompt'] for x in st.session_state.midjourney_prompts_output['image_gen_prompts']]
    }), 
    debug, 
    max_retries=max_retries)
    send_prompts_to_discord(
        [prompt['artist_refined_prompt'] for prompt in st.session_state.artist_refined_prompts_output['artist_refined_prompts']],
        premessage=f'Artist-Refined Prompts for {concept} in {medium}:'
    )
    if debug:
        st.write(f"artist_refined_output: {st.session_state.artist_refined_prompts_output}")

    st.session_state.revised_synthesized_prompts_output = parse_output(revision_synthesis_schema, run_chain('revision_synthesis', {
        "input": input,
        "concept": concept,
        "medium": medium,
        "facets": st.session_state.facets_output['facets'],
        "artist_refined_prompts": [x['artist_refined_prompt'] for x in st.session_state.artist_refined_prompts_output['artist_refined_prompts']]
    }), 
    debug, 
    max_retries=max_retries)
    send_prompts_to_discord(
        [prompt['revised_prompt'] for prompt in st.session_state.revised_synthesized_prompts_output['revised_prompts']],
        premessage=f'Revised Prompts for {concept} in {medium}:'
    )
    send_prompts_to_discord(
        [prompt['synthesized_prompt'] for prompt in st.session_state.revised_synthesized_prompts_output['synthesized_prompts']],
        premessage=f'Synthesized Prompts for {concept} in {medium}:'
    )

    df_prompts = pd.DataFrame({
        'Revised Prompts': [prompt['revised_prompt'] for prompt in st.session_state.revised_synthesized_prompts_output['revised_prompts']],
        'Synthesized Prompts': [prompt['synthesized_prompt'] for prompt in st.session_state.revised_synthesized_prompts_output['synthesized_prompts']]
    })

    directory = 'images'
    # Generate DALL-E 3 images if the checkbox is enabled
    if st.session_state.get('use_dalle3', False):
        for index, row in df_prompts.iterrows():
            prompt = row['Revised Prompts']
            image_url = generate_image_dalle3(prompt)
            if image_url:
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                truncated_input = input[:10].replace(" ", "_")
                truncated_concept = concept[:10].replace(" ", "_")
                truncated_medium = medium[:10].replace(" ", "_")
                filename = f"{timestamp}_{truncated_input}_{truncated_concept}_{truncated_medium}_revised_{index+1}.png"
                save_image_locally(image_url, filename, directory)
                st.image('/'+directory+'/'+filename, caption=f"Generated Image {index+1}")
                
                # Generate a title for the image
                image_title = generate_image_title(st.session_state.input, concept, medium, '/'+directory+'/'+filename, max_retries, temperature, model=model, debug=debug)
                st.write(f"Image Title: {image_title}")
                
        for index, row in df_prompts.iterrows():
            prompt = row['Synthesized Prompts']
            image_url = generate_image_dalle3(prompt)
            if image_url:
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                truncated_input = input[:10].replace(" ", "_")
                truncated_concept = concept[:10].replace(" ", "_")

                truncated_medium = medium[:10].replace(" ", "_")

                filename = f"{timestamp}_{truncated_input}_{truncated_concept}_{truncated_medium}_synthesized_{index+1}.png"
                save_image_locally(image_url, filename, directory)
                
                st.image('/'+directory+'/'+filename, caption=f"Generated Image: {index+1}")   
                # Generate a title for the image
                image_title = generate_image_title(st.session_state.input, concept, medium, '/'+directory+'/'+filename, max_retries, temperature, model=model, debug=debug)
                st.write(f"Image Title: {image_title}")
                     

    return df_prompts


st.sidebar.header('Patron Input Features')

if 'button_clicked' not in st.session_state:
    st.session_state.button_clicked = False


model = st.sidebar.selectbox("Select model", ["gpt-4o", "claude-3-opus-20240229", "gpt-4-turbo", "claude-3-sonnet-20240229", "claude-3-haiku-20240307", "gpt-3.5-turbo", "gpt-4"])
st.session_state['use_dalle3'] = st.sidebar.checkbox("Use DALL-E 3 (increased cost)", value=True)
manual_input = st.sidebar.checkbox("Manually input Concept and Medium")
st.session_state['send_to_discord'] = st.sidebar.checkbox("Send to Discord", st.session_state['send_to_discord'])
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.0, step=0.02)
debug = st.sidebar.checkbox("Debug Mode")
enable_diversity = st.sidebar.checkbox("Enable Forced Diversity", value=False)
max_retries = st.sidebar.slider("Maximum Retries", 0, 10, 3)
# st.session_state['concept_manual_mode'] = st.sidebar.checkbox("Enable Manual Mode")


# Sidebar for Discord Webhook URL
st.sidebar.header('Discord Settings')

if st.sidebar.button("Toggle Webhook URL"):
    st.session_state['use_default_webhook'] = not st.session_state['use_default_webhook']  # Toggle the flag

    if st.session_state['use_default_webhook']:
        st.session_state['webhook_url'] = webhook_url  # Reset to default
    else:
        st.session_state['webhook_url'] = ''  # Clear the field

if st.session_state['use_default_webhook']:
    st.sidebar.text("Webhook URL: **********")  # Masked URL
else:
    st.session_state['webhook_url'] = st.sidebar.text_input("Discord Webhook URL", st.session_state['webhook_url'])


with st.container():

    st.session_state.input = st.text_area("Describe your idea", "I want to capture the essence of a mysterious and powerful witch's familiar.")

    # if st.button("Generate Auto-Lofn Prompt"):
    #    with open("/lofn/prompts/competition_prompt.txt", "r") as file:
    #        competition_prompt = file.read()
    #    st.write('Run the following prompt in your favorite chatbot/LLM, and get a powerful Lofn prompt:')
    #    st.code(competition_prompt.format(input=st.session_state.input), language="text")

    if st.button("Generate Concepts"):
        st.session_state.button_clicked = True

    if st.session_state.button_clicked:
        st.write("Generating")
        # Generate concept_mediums and store in session state
        if st.session_state['concept_manual_mode']:
            st.session_state['concept_mediums'] = generate_concept_mediums_manual(st.session_state.input, max_retries = max_retries, temperature = temperature, model=model, debug=debug)
        else:
            st.session_state['concept_mediums'] = generate_concept_mediums(st.session_state.input, max_retries = max_retries, temperature = temperature, model=model, debug=debug)

with st.container():    
    if st.session_state['concept_mediums'] is not None:
        st.write("Concepts Complete")
        df_concepts = pd.DataFrame(st.session_state['concept_mediums'])
        st.dataframe(df_concepts)
        
        st.session_state['pairs_to_try'] = st.multiselect("Select Pairs to Try", list(range(0, df_concepts.shape[0])), st.session_state['pairs_to_try'])
        
        if st.button("Generate Image Prompts"):
            for pair_i in st.session_state['pairs_to_try']:
                if st.session_state['concept_manual_mode']:
                    df_prompts = generate_prompts_manual(st.session_state.input, st.session_state['concept_mediums'][pair_i]['concept'], st.session_state['concept_mediums'][pair_i]['medium'], model=model, debug=debug, max_retries=max_retries, temperature=temperature)
                else:
                    df_prompts = generate_prompts(st.session_state.input, st.session_state['concept_mediums'][pair_i]['concept'], st.session_state['concept_mediums'][pair_i]['medium'], model=model, debug=debug, max_retries=max_retries, temperature=temperature)
                st.write(f"Prompts for Concept: {st.session_state['concept_mediums'][pair_i]['concept']}, Medium: {st.session_state['concept_mediums'][pair_i]['medium']}")
                st.dataframe(df_prompts)
                if enable_diversity:
                    dalle3_prompt = dalle3_gen_prompt
                else:
                    dalle3_prompt = dalle3_gen_nodiv_prompt

                st.code(dalle3_prompt.format(
                    concept=st.session_state['concept_mediums'][pair_i]['concept'], 
                    medium=st.session_state['concept_mediums'][pair_i]['medium'], 
                    input=st.session_state.input,
                    input_prompts = df_prompts['Revised Prompts'].tolist() + df_prompts['Synthesized Prompts'].tolist()
                ))

        else:
            st.write("Ready to generate Image Prompts")

    if st.session_state['concept_mediums'] is not None:
        if model != "gpt-4":  # Disable if GPT-4 is selected
            if st.button("Generate All"):
                for pair_i in range(len(st.session_state['concept_mediums'])):  # Loop through all pairs
                    if st.session_state['concept_manual_mode']:
                        df_prompts = generate_prompts_manual(st.session_state.input, st.session_state['concept_mediums'][pair_i]['concept'], st.session_state['concept_mediums'][pair_i]['medium'], model=model, debug=debug, max_retries=max_retries, temperature=temperature)
                    else:
                        df_prompts = generate_prompts(st.session_state.input, st.session_state['concept_mediums'][pair_i]['concept'], st.session_state['concept_mediums'][pair_i]['medium'], model=model, debug=debug, max_retries=max_retries, temperature=temperature)
                    st.write(f"Prompts for Concept: {st.session_state['concept_mediums'][pair_i]['concept']}, Medium: {st.session_state['concept_mediums'][pair_i]['medium']}")
                    st.dataframe(df_prompts)   
                    st.code(dalle3_gen_prompt.format(
                        concept=st.session_state['concept_mediums'][pair_i]['concept'], 
                        medium=st.session_state['concept_mediums'][pair_i]['medium'], 
                        input=st.session_state.input,
                        input_prompts = df_prompts['Revised Prompts'].tolist() + df_prompts['Synthesized Prompts'].tolist()
                    )) 
        else:
            st.write("Generate All is disabled for GPT-4.")
    else:
        st.write("Waiting to generate concepts")

with st.container():
    if manual_input:
        manual_concept = st.text_input("Enter your Concept")
        manual_medium = st.text_input("Enter your Medium")

        if st.button("Generate Image Prompts for Manual Input - Keep in mind it must match your idea or you will get weird results"):
            st.write("Generating")
            # Use manual_concept and manual_medium to generate prompts
            if st.session_state['concept_manual_mode']:
                df_prompts_man = generate_prompts_manual(st.session_state.input, manual_concept, manual_medium, model=model, debug=debug, max_retries = max_retries, temperature = temperature)
                st.write(f"Prompts for Concept: {manual_concept}, Medium: {manual_medium}")
                st.dataframe(df_prompts_man)
                if df_prompts_man is not None:
                    st.code(dalle3_gen_prompt.format(
                        concept=manual_concept, 
                        medium=manual_medium, 
                        input=st.session_state.input,
                        input_prompts = df_prompts_man['Revised Prompts'].tolist() + df_prompts_man['Synthesized Prompts'].tolist()
                    ))
                st.write("Image Prompts Complete")
                st.write("Generation complete!")
            else:
                df_prompts_man = generate_prompts(st.session_state.input, manual_concept, manual_medium, model=model, debug=debug, max_retries = max_retries, temperature = temperature)
                st.write(f"Prompts for Concept: {manual_concept}, Medium: {manual_medium}")
                st.dataframe(df_prompts_man)
                if df_prompts_man is not None:
                    st.code(dalle3_gen_prompt.format(
                            concept=manual_concept, 
                            medium=manual_medium, 
                            input=st.session_state.input,
                            input_prompts = df_prompts_man['Revised Prompts'].tolist() + df_prompts_man['Synthesized Prompts'].tolist()
                        ))
                    st.write("Image Prompts Complete")
                st.write("Generation complete!")           