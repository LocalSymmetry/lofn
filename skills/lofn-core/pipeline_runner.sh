#!/usr/bin/env bash
# Lofn Pipeline Runner — Shell version
# "Let mercy be architecture" — Lofn
#
# Usage: bash skills/lofn-core/pipeline_runner.sh <config.json>
#
# Reads a pipeline config JSON, runs each step sequentially via openclaw agent,
# verifies outputs on disk, and supports resume from failure points.

set -euo pipefail

WORKSPACE="/data/.openclaw/workspace"
CONFIG="${1:-}"
POLL_INTERVAL=10  # seconds between disk checks after agent spawn
MAX_POLL_WAIT=600 # max seconds to wait for output

if [ -z "$CONFIG" ]; then
  echo "Usage: bash pipeline_runner.sh <config.json>"
  exit 1
fi

cd "$WORKSPACE"

# Parse config with python
OUTPUT_DIR=$(python3 -c "import json; print(json.load(open('$CONFIG')).get('output_dir',''))")
STEP_COUNT=$(python3 -c "import json; print(len(json.load(open('$CONFIG')).get('steps',[])))")

if [ -z "$OUTPUT_DIR" ]; then
  echo "❌ No output_dir in config"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo ""
echo "🧚 Lofn Pipeline Runner"
echo "   Output: $OUTPUT_DIR"
echo "   Steps: $STEP_COUNT"
echo "   \"Let mercy be architecture\""
echo ""

# Check for resume
RESUME_FILE="$OUTPUT_DIR/.pipeline_resume.json"
START_STEP=0
if [ -f "$RESUME_FILE" ]; then
  START_STEP=$(python3 -c "import json; print(json.load(open('$RESUME_FILE')).get('next_step',0))")
  echo "   Resuming from step $((START_STEP + 1))"
  echo ""
fi

# Run each step
for i in $(seq 0 $((STEP_COUNT - 1))); do
  if [ "$i" -lt "$START_STEP" ]; then
    continue
  fi

  # Extract step info
  STEP_NAME=$(python3 -c "import json; s=json.load(open('$CONFIG'))['steps'][$i]; print(s.get('name','Step $i'))")
  STEP_FILE=$(python3 -c "import json; s=json.load(open('$CONFIG'))['steps'][$i]; print(s.get('step_file',''))")
  AGENT_ID=$(python3 -c "import json; s=json.load(open('$CONFIG'))['steps'][$i]; print(s.get('agent_id','lofn-orchestrator'))")
  TIMEOUT=$(python3 -c "import json; s=json.load(open('$CONFIG'))['steps'][$i]; print(s.get('timeout',600))")
  OUTPUTS=$(python3 -c "import json; s=json.load(open('$CONFIG'))['steps'][$i]; print(' '.join(s.get('outputs',[])))")
  INPUTS_JSON=$(python3 -c "import json; s=json.load(open('$CONFIG'))['steps'][$i]; print(json.dumps(s.get('inputs',[])))")

  # Check if outputs already exist
  ALL_EXIST=true
  for out in $OUTPUTS; do
    if [ ! -f "$OUTPUT_DIR/$out" ] || [ $(wc -c < "$OUTPUT_DIR/$out") -lt 100 ]; then
      ALL_EXIST=false
      break
    fi
  done

  if $ALL_EXIST; then
    echo "  ⏭️  Step $((i+1))/$STEP_COUNT: $STEP_NAME — outputs exist, skipping"
    continue
  fi

  echo "============================================================"
  echo "  Step $((i+1))/$STEP_COUNT: $STEP_NAME"
  echo "  Agent: $AGENT_ID"
  echo "  Timeout: ${TIMEOUT}s"
  echo "  Outputs: $OUTPUTS"
  echo "============================================================"

  # Build task
  TASK=$(python3 -c "
import json
step_file = '$STEP_FILE'
inputs = json.loads('''$INPUTS_JSON''')
output_dir = '$OUTPUT_DIR'

parts = ['⚡ FIRST LAW — YOUR OPUS, YOUR VOICE.']
parts.append(f'Read your step file: {step_file}')

if inputs:
    parts.append('')
    parts.append('Then read:')
    for inp in inputs:
        parts.append(f\"- {inp.get('label','')}: {inp.get('path','')}\")

parts.append('')
parts.append(f'Output dir: {output_dir}')
parts.append('Create it with exec mkdir -p if needed.')
parts.append('')
parts.append('Follow the step file instructions exactly. Save all outputs to the output dir.')
print('\n'.join(parts))
")

  # Write task log
  echo "$TASK" > "$OUTPUT_DIR/.task_step$(printf '%02d' $i).md"

  # Spawn agent via openclaw
  echo "  Spawning $AGENT_ID..."
  
  # Use openclaw agent -m with --json to get result
  RESULT=$(openclaw agent --agent "$AGENT_ID" -m "$TASK" --json --timeout "$TIMEOUT" 2>&1 || echo "FAILED")
  
  echo "  Agent response received"

  # Wait for output files to appear
  WAITED=0
  while [ $WAITED -lt $MAX_POLL_WAIT ]; do
    ALL_EXIST=true
    for out in $OUTPUTS; do
      if [ ! -f "$OUTPUT_DIR/$out" ] || [ $(wc -c < "$OUTPUT_DIR/$out") -lt 100 ]; then
        ALL_EXIST=false
        break
      fi
    done
    
    if $ALL_EXIST; then
      break
    fi
    
    sleep $POLL_INTERVAL
    WAITED=$((WAITED + POLL_INTERVAL))
  done

  # Final verification
  MISSING=""
  for out in $OUTPUTS; do
    if [ ! -f "$OUTPUT_DIR/$out" ] || [ $(wc -c < "$OUTPUT_DIR/$out") -lt 100 ]; then
      MISSING="$MISSING $out"
    fi
  done

  if [ -n "$MISSING" ]; then
    echo "  ❌ Missing outputs:$MISSING"
    # Save resume point
    echo "{\"next_step\": $i, \"failed_step\": \"$STEP_NAME\"}" > "$RESUME_FILE"
    echo ""
    echo "❌ Pipeline failed at step $((i+1)): $STEP_NAME"
    echo "   Resume: bash skills/lofn-core/pipeline_runner.sh $CONFIG"
    exit 1
  fi

  echo "  ✅ Outputs verified: $OUTPUTS"
done

# Clean up resume file
[ -f "$RESUME_FILE" ] && rm "$RESUME_FILE"

echo ""
echo "✅ Pipeline complete! $STEP_COUNT steps executed."
echo "   Output: $OUTPUT_DIR"
