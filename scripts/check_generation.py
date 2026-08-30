import re
import sys

def analyze_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    variations = re.split(r'## VARIATION', content)[1:]
    print(f"Found {len(variations)} variations.")
    
    for i, var in enumerate(variations):
        lines = var.strip().split('\n')
        title_line = lines[0]
        title = title_line.replace('—', '').strip()
        
        # Extract prompt
        prompt_match = re.search(r'### SUNO STYLE PROMPT\s*\n+([^#\n][^#]*)', var, re.DOTALL)
        if prompt_match:
            prompt = prompt_match.group(1).strip()
            prompt_len = len(prompt)
            print(f"Var {i+1} ({title}): Prompt length: {prompt_len} chars")
        else:
            print(f"Var {i+1} ({title}): Prompt not found!")
            
        # Extract lyrics
        lyrics_match = re.search(r'### SUNO LYRICS\s*\n+(.*)', var, re.DOTALL)
        if lyrics_match:
            lyrics_text = lyrics_match.group(1).strip()
            lyric_lines = [l.strip() for l in lyrics_text.split('\n') if l.strip()]
            
            # Count sung lines (lines that don't start with [ and are not empty)
            sung_lines = [l for l in lyric_lines if not l.startswith('[') and l.strip()]
            print(f"Var {i+1} ({title}): Total sung lines: {len(sung_lines)}")
            
            # Check for Theme and SONG FORM
            has_theme = any('[Theme:' in l for l in lyric_lines[:3])
            has_form = any('[SONG FORM:' in l for l in lyric_lines[:3])
            print(f"Var {i+1} ({title}): Has [Theme]: {has_theme}, Has [SONG FORM]: {has_form}")
            
            # Check for EMO headers
            emo_headers = [l for l in lyric_lines if l.startswith('[') and 'EMO:' in l]
            print(f"Var {i+1} ({title}): EMO headers found: {len(emo_headers)}")
            
            # Check for SFX cue
            sfx_cues = [l for l in lyric_lines if '[SFX:' in l or 'SFX:' in l]
            print(f"Var {i+1} ({title}): SFX cues found: {len(sfx_cues)}")
        else:
            print(f"Var {i+1} ({title}): Lyrics not found!")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        analyze_file(sys.argv[1])
