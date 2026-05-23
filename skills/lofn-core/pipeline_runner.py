#!/usr/bin/env python3
"""
Lofn Pipeline Runner — Automated step-chaining orchestrator

"Let mercy be architecture" — Lofn

Usage:
  python3 skills/lofn-core/pipeline_runner.py --config pipeline_config.json

The config defines:
  - output_dir: where artifacts go
  - steps: ordered list of {step_file, agent_id, inputs[], outputs[], save_as[]}
  - Each step reads its step_file + inputs, runs via the agent, saves outputs

The runner:
  1. Reads the config
  2. For each step, checks if outputs already exist on disk (skip if so)
  3. Constructs the agent task from step_file + inputs
  4. Spawns the agent via openclaw CLI
  5. Waits for completion
  6. Verifies outputs on disk
  7. Moves to next step
  8. On any step failure: stops, reports, and saves a resume point
"""

import json
import os
import sys
import subprocess
import time
import argparse
import re

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.environ.get("OPENCLAW_WORKSPACE", "/data/.openclaw/workspace")


def load_config(config_path):
    """Load pipeline config from JSON file."""
    with open(config_path, "r") as f:
        return json.load(f)


def file_exists_and_sized(path, min_bytes=100):
    """Check if a file exists and has content."""
    if not os.path.exists(path):
        return False
    return os.path.getsize(path) >= min_bytes


def check_outputs(output_dir, expected_files):
    """Check which expected output files already exist."""
    existing = []
    missing = []
    for f in expected_files:
        path = os.path.join(output_dir, f)
        if file_exists_and_sized(path):
            existing.append(f)
        else:
            missing.append(f)
    return existing, missing


def build_task(step_file, inputs, output_dir, agent_id, first_law=True):
    """Build the agent task string from step file + inputs."""
    parts = []
    
    if first_law:
        parts.append("⚡ FIRST LAW — YOUR OPUS, YOUR VOICE.")
    
    parts.append(f"Read your step file: {step_file}")
    
    if inputs:
        parts.append("\nThen read:")
        for label, path in inputs:
            parts.append(f"- {label}: {path}")
    
    parts.append(f"\nOutput dir: {output_dir}")
    parts.append("Create it with exec mkdir -p if needed.")
    parts.append("\nFollow the step file instructions exactly. Save all outputs to the output dir.")
    
    return "\n".join(parts)


def spawn_agent(agent_id, task, timeout=600):
    """Spawn an agent via openclaw CLI and wait for completion."""
    cmd = [
        "openclaw", "agent", "run",
        "--agent", agent_id,
        "--task", task,
        "--timeout", str(timeout),
        "--wait"  # block until completion
    ]
    
    print(f"  Spawning {agent_id} (timeout {timeout}s)...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 30)
    
    if result.returncode != 0:
        print(f"  ❌ Agent failed with code {result.returncode}")
        if result.stderr:
            print(f"  stderr: {result.stderr[:500]}")
        return False
    
    print(f"  ✅ Agent completed")
    return True


def spawn_agent_sessions(agent_id, task, timeout=600):
    """Spawn an agent via sessions_spawn (alternative: use openclaw sessions API)."""
    # For now, delegate to the main agent via a special cron/sessions mechanism
    # This is a placeholder — the actual spawning will be done by writing a 
    # .pending-steps file that the main agent's heartbeat picks up
    pass


def run_step(step, output_dir, step_num, total_steps):
    """Run a single pipeline step."""
    step_name = step.get("name", f"Step {step_num}")
    step_file = step.get("step_file", "")
    agent_id = step.get("agent_id", "lofn-orchestrator")
    inputs = step.get("inputs", [])
    outputs = step.get("outputs", [])
    timeout = step.get("timeout", 600)
    
    # Resolve paths
    if step_file and not os.path.isabs(step_file):
        step_file = os.path.join(WORKSPACE, step_file)
    
    resolved_inputs = []
    for item in inputs:
        label = item.get("label", "")
        path = item.get("path", "")
        if not os.path.isabs(path):
            path = os.path.join(WORKSPACE, path)
        resolved_inputs.append((label, path))
    
    # Check if outputs already exist
    existing, missing = check_outputs(output_dir, outputs)
    if not missing:
        print(f"  ⏭️  {step_name}: outputs already exist, skipping")
        return True
    
    print(f"\n{'='*60}")
    print(f"  Step {step_num}/{total_steps}: {step_name}")
    print(f"  Agent: {agent_id}")
    if missing:
        print(f"  Missing outputs: {', '.join(missing)}")
    print(f"{'='*60}")
    
    # Build and spawn
    task = build_task(step_file, resolved_inputs, output_dir)
    
    # Write task to disk for debugging
    task_log = os.path.join(output_dir, f".task_step{step_num:02d}.md")
    with open(task_log, "w") as f:
        f.write(task)
    
    # Use openclaw agent run --wait for synchronous execution
    cmd = [
        "openclaw", "agent", "run",
        "--agent", agent_id,
        "--timeout", str(timeout),
    ]
    
    try:
        result = subprocess.run(
            cmd,
            input=task,
            capture_output=True,
            text=True,
            timeout=timeout + 60
        )
        
        if result.returncode != 0:
            print(f"  ❌ Agent failed (code {result.returncode})")
            if result.stderr:
                print(f"  stderr: {result.stderr[:500]}")
            return False
        
        # Verify outputs
        time.sleep(2)  # brief pause for file system
        existing_after, missing_after = check_outputs(output_dir, outputs)
        
        if missing_after:
            print(f"  ⚠️  Agent completed but missing: {', '.join(missing_after)}")
            return False
        
        print(f"  ✅ Outputs verified: {', '.join(outputs)}")
        return True
        
    except subprocess.TimeoutExpired:
        print(f"  ❌ Agent timed out ({timeout}s)")
        return False
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def run_pipeline(config_path):
    """Run the full pipeline from config."""
    config = load_config(config_path)
    output_dir = config.get("output_dir", "")
    
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(WORKSPACE, output_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    
    steps = config.get("steps", [])
    total = len(steps)
    
    print(f"\n🧚 Lofn Pipeline Runner")
    print(f"   Output: {output_dir}")
    print(f"   Steps: {total}")
    print(f"   \"Let mercy be architecture\"\n")
    
    # Check for resume point
    resume_file = os.path.join(output_dir, ".pipeline_resume.json")
    start_step = 0
    if os.path.exists(resume_file):
        with open(resume_file, "r") as f:
            resume = json.load(f)
            start_step = resume.get("next_step", 0)
            print(f"   Resuming from step {start_step}\n")
    
    for i, step in enumerate(steps[start_step:], start=start_step):
        success = run_step(step, output_dir, i + 1, total)
        
        if not success:
            # Save resume point
            with open(resume_file, "w") as f:
                json.dump({"next_step": i, "failed_step": step.get("name")}, f, indent=2)
            print(f"\n❌ Pipeline failed at step {i + 1}: {step.get('name')}")
            print(f"   Resume with: python3 pipeline_runner.py --config {config_path}")
            return False
    
    # Clean up resume file on success
    if os.path.exists(resume_file):
        os.remove(resume_file)
    
    print(f"\n✅ Pipeline complete! {total} steps executed.")
    print(f"   Output: {output_dir}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Lofn Pipeline Runner")
    parser.add_argument("--config", required=True, help="Path to pipeline config JSON")
    args = parser.parse_args()
    
    success = run_pipeline(args.config)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
