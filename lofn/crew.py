from crewai import Crew, Process
from lofn.tasks import (
    essence_and_facets_task,
    concepts_task,
    artist_and_refined_concepts_task,
    medium_task,
    refine_medium_task,
    shuffled_review_task
)

# Form the crew
crew = Crew(
    agents=[
        essence_and_facets_agent,
        concepts_agent,
        artist_and_refined_concepts_agent,
        medium_agent,
        refine_medium_agent,
        shuffled_review_agent
    ],
    tasks=[
        essence_and_facets_task,
        concepts_task,
        artist_and_refined_concepts_task,
        medium_task,
        refine_medium_task,
        shuffled_review_task
    ],
    process=Process.sequential,
    memory=True,
    cache=True,
    max_rpm=100,
    share_crew=True
)
