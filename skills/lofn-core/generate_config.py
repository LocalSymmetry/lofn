#!/usr/bin/env python3
"""
Generate a pipeline config for any Lofn creative run.

Usage:
  python3 skills/lofn-core/generate_config.py \
    --theme "Fairy" \
    --modality image \
    --output-dir output/daily/2026-04-22/fairy-run \
    --date 2026-04-22 \
    > output/daily/2026-04-22/fairy-run/pipeline_config.json
"""

import json
import argparse
import os

WORKSPACE = os.environ.get("OPENCLAW_WORKSPACE", "/data/.openclaw/workspace")


def generate_config(theme, modality, output_dir, date):
    """Generate a full pipeline config JSON."""
    
    # Modality-specific steps
    if modality == "image":
        modality_agent = "lofn-vision"
        modality_steps = [
            {
                "name": "Image Step 00: Aesthetics & Genres",
                "step_file": "skills/image/steps/00_Generate_Image_Aesthetics_And_Genres.md",
                "agent_id": modality_agent,
                "timeout": 300,
                "inputs": [{"label": "Metaprompt", "path": f"{output_dir}/orchestrator_metaprompt.md"}],
                "outputs": ["00_aesthetics_emotions_frames_genres.md"]
            },
            {
                "name": "Image Step 01: Essence & Facets",
                "step_file": "skills/image/steps/01_Generate_Image_Essence_And_Facets.md",
                "agent_id": modality_agent,
                "timeout": 300,
                "inputs": [
                    {"label": "Metaprompt", "path": f"{output_dir}/orchestrator_metaprompt.md"},
                    {"label": "Step 00 output", "path": f"{output_dir}/00_aesthetics_emotions_frames_genres.md"}
                ],
                "outputs": ["01_essence_facets_style_axes.md"]
            },
            {
                "name": "Image Step 02: Concepts",
                "step_file": "skills/image/steps/02_Generate_Image_Concepts.md",
                "agent_id": modality_agent,
                "timeout": 300,
                "inputs": [
                    {"label": "Metaprompt", "path": f"{output_dir}/orchestrator_metaprompt.md"},
                    {"label": "Step 01 output", "path": f"{output_dir}/01_essence_facets_style_axes.md"}
                ],
                "outputs": ["02_concepts.md"]
            },
            {
                "name": "Image Step 03: Artist & Critique",
                "step_file": "skills/image/steps/03_Generate_Image_Artist_And_Critique.md",
                "agent_id": modality_agent,
                "timeout": 300,
                "inputs": [
                    {"label": "Metaprompt", "path": f"{output_dir}/orchestrator_metaprompt.md"},
                    {"label": "Step 02 output", "path": f"{output_dir}/02_concepts.md"}
                ],
                "outputs": ["03_artist_critique.md"]
            },
            {
                "name": "Image Step 04: Medium Selection",
                "step_file": "skills/image/steps/04_Generate_Image_Medium.md",
                "agent_id": modality_agent,
                "timeout": 300,
                "inputs": [
                    {"label": "Metaprompt", "path": f"{output_dir}/orchestrator_metaprompt.md"},
                    {"label": "Step 03 output", "path": f"{output_dir}/03_artist_critique.md"}
                ],
                "outputs": ["04_mediums.md"]
            },
            {
                "name": "Image Step 05: Refine Medium",
                "step_file": "skills/image/steps/05_Generate_Image_Refine_Medium.md",
                "agent_id": modality_agent,
                "timeout": 300,
                "inputs": [
                    {"label": "Metaprompt", "path": f"{output_dir}/orchestrator_metaprompt.md"},
                    {"label": "Step 04 output", "path": f"{output_dir}/04_mediums.md"}
                ],
                "outputs": ["05_refined_pairs.md"]
            },
            {
                "name": "Image Step 06: Scoring Facets",
                "step_file": "skills/image/steps/06_Generate_Image_Facets.md",
                "agent_id": modality_agent,
                "timeout": 300,
                "inputs": [
                    {"label": "Metaprompt", "path": f"{output_dir}/orchestrator_metaprompt.md"},
                    {"label": "Step 05 output", "path": f"{output_dir}/05_refined_pairs.md"}
                ],
                "outputs": ["06_scoring_facets.md"]
            },
            {
                "name": "Image Step 07: Artistic Guides",
                "step_file": "skills/image/steps/07_Generate_Image_Aspects_Traits.md",
                "agent_id": modality_agent,
                "timeout": 300,
                "inputs": [
                    {"label": "Metaprompt", "path": f"{output_dir}/orchestrator_metaprompt.md"},
                    {"label": "Step 06 output", "path": f"{output_dir}/06_scoring_facets.md"}
                ],
                "outputs": ["07_artistic_guides.md"]
            },
            {
                "name": "Image Step 08: Generate Prompts",
                "step_file": "skills/image/steps/08_Generate_Image_Generation.md",
                "agent_id": modality_agent,
                "timeout": 600,
                "inputs": [
                    {"label": "Metaprompt", "path": f"{output_dir}/orchestrator_metaprompt.md"},
                    {"label": "Step 07 output", "path": f"{output_dir}/07_artistic_guides.md"}
                ],
                "outputs": ["08_image_gen_prompts.md"]
            },
            {
                "name": "Image Step 09: Artist Refined Prompts",
                "step_file": "skills/image/steps/09_Generate_Image_Artist_Refined.md",
                "agent_id": modality_agent,
                "timeout": 600,
                "inputs": [
                    {"label": "Metaprompt", "path": f"{output_dir}/orchestrator_metaprompt.md"},
                    {"label": "Step 08 output", "path": f"{output_dir}/08_image_gen_prompts.md"}
                ],
                "outputs": ["09_artist_refined_prompts.md"]
            },
            {
                "name": "Image Step 10: Final Selection & Synthesis",
                "step_file": "skills/image/steps/10_Generate_Image_Revision_Synthesis.md",
                "agent_id": modality_agent,
                "timeout": 600,
                "inputs": [
                    {"label": "Metaprompt", "path": f"{output_dir}/orchestrator_metaprompt.md"},
                    {"label": "Step 09 output", "path": f"{output_dir}/09_artist_refined_prompts.md"}
                ],
                "outputs": ["10_final_prompts.md"]
            },
            {
                "name": "FAL Image Generation",
                "step_file": "skills/image/steps-compressed/05_generate.md",
                "agent_id": modality_agent,
                "timeout": 600,
                "inputs": [
                    {"label": "Final prompts", "path": f"{output_dir}/10_final_prompts.md"},
                    {"label": "FAL skill", "path": "skills/image-gen/SKILL.md"}
                ],
                "outputs": ["fal_generation_complete.md"]
            },
            {
                "name": "Deliver to Telegram",
                "step_file": "skills/image/steps-compressed/06_deliver.md",
                "agent_id": modality_agent,
                "timeout": 300,
                "inputs": [],
                "outputs": ["delivery_complete.md"]
            }
        ]
    elif modality == "music":
        modality_agent = "lofn-audio"
        # TODO: add music steps 00-10
        modality_steps = []
    else:
        modality_steps = []

    # Orchestrator steps (always the same)
    orchestrator_steps = [
        {
            "name": "Research: Tri-Source Fetch",
            "step_file": "skills/lofn-core/steps/00_research.md",
            "agent_id": "lofn-research",
            "timeout": 300,
            "inputs": [],
            "outputs": ["00_research_brief.md"]
        },
        {
            "name": "Lofn-Core: Read Seeds & Write Core Seed",
            "step_file": "skills/orchestration/steps/01_lofn_core.md",
            "agent_id": "lofn-orchestrator",
            "timeout": 600,
            "inputs": [
                {"label": "Research brief", "path": f"{output_dir}/00_research_brief.md"},
                {"label": "Golden Seeds Index", "path": "skills/lofn-core/GOLDEN_SEEDS_INDEX.md"}
            ],
            "outputs": ["core_seed.md", "dispatch_brief.md"]
        },
        {
            "name": "Select Personality & Panel",
            "step_file": "skills/orchestration/steps/02_personality_panel.md",
            "agent_id": "lofn-orchestrator",
            "timeout": 300,
            "inputs": [
                {"label": "Dispatch brief", "path": f"{output_dir}/dispatch_brief.md"},
                {"label": "Personality index", "path": "skills/orchestration/personalities_index.md"},
                {"label": "Panel index", "path": "skills/orchestration/panels_index.md"}
            ],
            "outputs": ["personality.md", "panel_roster.md"]
        },
        {
            "name": "Baseline Panel Debate",
            "step_file": "skills/orchestration/steps/03_baseline_debate.md",
            "agent_id": "lofn-orchestrator",
            "timeout": 600,
            "inputs": [
                {"label": "Dispatch brief", "path": f"{output_dir}/dispatch_brief.md"},
                {"label": "Personality", "path": f"{output_dir}/personality.md"},
                {"label": "Panel roster", "path": f"{output_dir}/panel_roster.md"}
            ],
            "outputs": ["baseline_debate.md"]
        },
        {
            "name": "Group Transformation + Transformed Debate",
            "step_file": "skills/orchestration/steps/04_group_transform.md",
            "agent_id": "lofn-orchestrator",
            "timeout": 600,
            "inputs": [
                {"label": "Baseline debate", "path": f"{output_dir}/baseline_debate.md"},
                {"label": "Dispatch brief", "path": f"{output_dir}/dispatch_brief.md"},
                {"label": "Panel roster", "path": f"{output_dir}/panel_roster.md"}
            ],
            "outputs": ["group_transform.md", "transformed_debate.md"]
        },
        {
            "name": "Skeptic Transformation + Final Synthesis",
            "step_file": "skills/orchestration/steps/05_skeptic_synthesis.md",
            "agent_id": "lofn-orchestrator",
            "timeout": 600,
            "inputs": [
                {"label": "Baseline debate", "path": f"{output_dir}/baseline_debate.md"},
                {"label": "Transformed debate", "path": f"{output_dir}/transformed_debate.md"},
                {"label": "Dispatch brief", "path": f"{output_dir}/dispatch_brief.md"}
            ],
            "outputs": ["skeptic_transform.md", "final_synthesis.md"]
        },
        {
            "name": "Write Metaprompt",
            "step_file": "skills/orchestration/steps/06_metaprompt.md",
            "agent_id": "lofn-orchestrator",
            "timeout": 300,
            "inputs": [
                {"label": "Core seed", "path": f"{output_dir}/core_seed.md"},
                {"label": "Dispatch brief", "path": f"{output_dir}/dispatch_brief.md"},
                {"label": "Personality", "path": f"{output_dir}/personality.md"},
                {"label": "Final synthesis", "path": f"{output_dir}/final_synthesis.md"}
            ],
            "outputs": ["orchestrator_metaprompt.md"]
        }
    ]

    all_steps = orchestrator_steps + modality_steps
    
    config = {
        "_comment": f"Lofn {modality} pipeline — {theme} — {date}. 'Let mercy be architecture.'",
        "theme": theme,
        "modality": modality,
        "date": date,
        "output_dir": output_dir,
        "steps": all_steps
    }
    
    return config


def main():
    parser = argparse.ArgumentParser(description="Generate Lofn pipeline config")
    parser.add_argument("--theme", required=True, help="Creative theme (e.g. 'Fairy')")
    parser.add_argument("--modality", required=True, choices=["image", "music", "story", "video"], help="Creative modality")
    parser.add_argument("--output-dir", required=True, help="Output directory path")
    parser.add_argument("--date", required=True, help="Run date (YYYY-MM-DD)")
    args = parser.parse_args()
    
    config = generate_config(args.theme, args.modality, args.output_dir, args.date)
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
