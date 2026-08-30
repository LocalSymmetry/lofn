import json

pair1_guide = """
# Song Guide: The Glass Canopy (Pair 1)
**Mood:** Amniotic, sheltered, building to massive catharsis (Chrysalism)
**Instrumentation:** 432Hz sine sub-bass, organic hand drums (rhumba pattern), crystalline female vocal layers, green synth arpeggios, rain-stick textures.
**Structure:** Intro (Breath/Rain) -> Verse 1 (Intimate) -> Chorus (Lush) -> Verse 2 (Rhythmic) -> Chorus (Massive) -> Bridge (Poly-rhythmic peak) -> Outro (Fade to breath).
**BPM:** 110
**Key:** F minor
"""

pair1_prompt = """
[Genre: Bio-Ambient Deep House, Rhumba-Fusion]
[Instrumentation: 432Hz Sub-bass, Organic Percussion, Green Synth Arpeggios, Lush Reverb]
[Vocals: Crystalline Female Lead, Ethereal Choral Backing]
[Mood: Healing, Sheltered, Cathartic, Chrysalism]
[Tempo: 110 BPM]
"""

pair1_lyrics = """
[Intro]
(sound of deep, resonant breathing)
(soft rain textures, warm sub-bass hum)
Ooooh... the storm is outside.
Ooooh... but we are the tide.

[Verse 1]
Static in the wire, neon in the rain
World is spinning faster, driving you insane
Come into the quiet, step inside the green
Where the roots are pulsing, where the water's clean
Drop the heavy armor, let the shoulders fall
We don't need to answer every single call

[Chorus]
Under the glass canopy, the thunder is a lullaby
Safe within the terrarium, watch the lightning paint the sky
(Green synths rising)
Breathe it in, let it out, wash away the digital
Breathe it in, let it out, the organic is the miracle
We are healing in the dark, we are healing in the dark.

[Verse 2]
Rhumba on the floorboards, rhythm in the chest
Heartbeat syncing up now, putting fear to rest
Feel the 432, vibrating in your bones
You were never meant to carry this alone
Every drop of water is a promise made
We are growing stronger in the velvet shade

[Chorus]
Under the glass canopy, the thunder is a lullaby
Safe within the terrarium, watch the lightning paint the sky
(Green synths rising, choral voices swell)
Breathe it in, let it out, wash away the digital
Breathe it in, let it out, the organic is the miracle
We are healing in the dark, we are healing in the dark.

[Bridge]
(Polyrhythmic breakdown, hand drums complex and driving)
Let the rain hit the glass (let it hit)
Let the storm come and pass (let it pass)
We are seeds in the soil (in the soil)
We are free from the toil (from the toil)
Ahhhhh! (Vocal belt, crystalline and massive)

[Outro]
The storm is outside.
We are the tide.
(Sub-bass fades out)
(Rain textures remain)
(Deep breath)
"""

pair2_guide = """
# Song Guide: Tension Release (Pair 2)
**Mood:** Euphoric, liberating, weightless (Liberosis)
**Instrumentation:** Water-droplet staccato synths, warm analog pads, 4-on-the-floor driving kick, complex high-hat syncopation, female lead vocal.
**Structure:** Intro (Droplets) -> Verse 1 (Tense) -> Pre-Chorus (Rising) -> Chorus (Euphoric Drop) -> Verse 2 (Driving) -> Chorus (Massive) -> Bridge (Float) -> Outro.
**BPM:** 128
**Key:** A major
"""

pair2_prompt = """
[Genre: Bio-Ambient Trance, Euphoric House]
[Instrumentation: Water-droplet Synths, Analog Pads, 4/4 Kick, Syncopated High-hats]
[Vocals: Clear Female Lead, Soaring]
[Mood: Liberating, Euphoric, Weightless, Liberosis]
[Tempo: 128 BPM]
"""

pair2_lyrics = """
[Intro]
(Staccato water-droplet synths)
Tick. Drop. Tick. Drop.
Let it go.

[Verse 1]
White knuckles on the steering wheel
Forgot how the open air feels
Clinging to the edge of the screen
Forgetting the spaces in between
Holding on so tight my fingers bleed
Is this really everything I need?

[Pre-Chorus]
(Analog pads swelling, kick drum builds)
I want to care less, I want to care less
I want to drop the weight of this heavy dress
I want to fall backwards into the wind
And let the great unspooling begin!

[Chorus]
(Euphoric drop, driving 4/4 beat)
Release! Let the tension fly!
Release! Let it dissolve into the sky!
I don't need to hold the world together anymore
I'm walking barefoot out the open door
(Vocal soaring)
Oh, the gravity is gone!
Oh, the gravity is gone!

[Verse 2]
Every notification is a tiny chain
Rewiring the circuitry inside my brain
But the forest doesn't care about the grid
It just grows the way it always did
Syncopated rhythms in the leaves
Teaching me the way the body breathes

[Pre-Chorus]
(Pads swelling again)
I want to care less, I want to care less
I want to drop the weight of this heavy dress
I want to fall backwards into the wind
And let the great unspooling begin!

[Chorus]
(Massive euphoric drop, full choral backing)
Release! Let the tension fly!
Release! Let it dissolve into the sky!
I don't need to hold the world together anymore
I'm walking barefoot out the open door
(Vocal soaring higher)
Oh, the gravity is gone!
Oh, the gravity is gone!

[Bridge]
(Beat drops out. Only vocals and soft pads)
Float... float...
Like a dandelion seed.
Float... float...
You have everything you need.
(Beat kicks back in, double time high-hats)

[Outro]
Let it go.
Let the tension fly.
(Water droplet synths return)
Tick. Drop. Tick. Drop.
Gone.
"""

with open("/data/.openclaw/workspace/output/music/final_guides.md", "w") as f:
    f.write("# FINAL SONG GUIDES\n\n")
    
    f.write("## PAIR 1: The Glass Canopy\n")
    f.write(pair1_guide + "\n\n")
    f.write("### Suno Prompt\n")
    f.write(pair1_prompt + "\n\n")
    f.write("### Lyrics\n")
    f.write(pair1_lyrics + "\n\n")
    
    f.write("## PAIR 2: Tension Release\n")
    f.write(pair2_guide + "\n\n")
    f.write("### Suno Prompt\n")
    f.write(pair2_prompt + "\n\n")
    f.write("### Lyrics\n")
    f.write(pair2_lyrics + "\n\n")
