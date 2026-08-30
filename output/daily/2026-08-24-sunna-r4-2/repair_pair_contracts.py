from __future__ import annotations

import json
from pathlib import Path


PATH = Path(__file__).resolve().parent / "music" / "concept_medium_pairs.json"
pairs = json.loads(PATH.read_text(encoding="utf-8-sig"))

voice = "Sunna: late-20s-to-early-30s adult female, light Scandinavian English accent"
verbs = ["reserve", "challenge", "permit", "season", "select", "declare"]
forms = [
    "linear reservation",
    "scorecard rounds",
    "stop/start antiphony",
    "cumulative call-response",
    "cut-up photo-strip rondeau",
    "question-to-declaration",
]
for pair, verb, form in zip(pairs, verbs, forms):
    pair["action_verb"] = verb
    pair["form_rule"] = form
    pair["vocal_fingerprint"] = (
        f"{voice}; bright chest-led mezzo, crisp consonants, smiling command catch; "
        "every final Style field must retain the age band, accent, two tone descriptors, "
        "and two unique phrasing traits"
    )

pairs[0]["medium"] = (
    "Lacquered guitar power pop at 116 BPM with melodic pick bass, candy-hard drums, "
    "tambourine, handclaps, compact girl-group thirds and Sunna: late-20s-to-early-30s "
    "adult female, light Scandinavian English accent, bright chest-led mezzo. A four-note "
    "whistle-guitar figure brackets the vocal. One clean two-tone upstroke replaces the verse "
    "guitar only when she crosses the diner threshold and the invitation gains its place."
)

p2 = pairs[1]
p2["seed_title"] = "MAKE IT NINE"
p2["concept"] = (
    "At the blue windmill hole, Sunna turns flirtation into a miniature-golf wager: beat her "
    "score and choose the next round. The blade interrupts her long-A hook once; the chosen "
    "adult counters with nine-oh-five and two blue balls, and their two rolls restore the meter "
    "before MAKE IT NINE returns."
)
p2["medium"] = (
    "Glossy dembow-pop scorecard rounds at 105 BPM with round synth bass doubled lightly by "
    "electric bass, crisp kick/snare, rim, guiro, muted guitar, bright claps and Sunna: "
    "late-20s-to-early-30s adult female, light Scandinavian English accent, conversational "
    "bright mezzo. A bass-plus-organ cell brackets the vocal. The moving blade cuts one bar "
    "short; two mutual ball rolls restore the missing beat. Rocksteady organ is the single seam."
)
p2["singer"]["physical_situation"] = (
    "She palms one blue miniature-golf ball at the moving windmill while the chosen adult marks "
    "a scorecard and holds the second ball."
)
p2["singer"]["unsaid_ache"] = (
    "She wants an equal who will play back, counter her time, and risk looking ridiculous beside her."
)
p2["surface"].update(
    dominant_body_image="Her blue ball waits for the blade, then two balls roll through on the beat the blade had stolen.",
    plain_language_ache="Challenge me, choose the next round, and mean it.",
    candidate_hooks=["Make It Nine", "Nine-Oh-Five, Your Turn", "Blue Ball, Beat My Score", "Two Rolls Restore It"],
    panel_lean_hook="Make It Nine",
)
p2["golden_seed_alloy"].update(
    mythic_material_escalation=(
        "One blue ball in a palm becomes a windmill blade cutting the song short; two blue balls "
        "rolling together restore the missing beat and return to their laughing hands."
    ),
    exact_detail="Nine is the game score and nine-oh-five the counter-time; their different functions must stay audible.",
    controlled_fracture="The windmill blade removes one beat from the long-A hook once; two mutual rolls restore the full bar.",
    routing_destination="Score and blue ball to rounds; blade-cut meter to middle hinge; two-roll restoration to final return; current-event pressure to sidecar only.",
)
p2["triple_arch_ritual_paragraph"] = (
    "One blue ball waits in Sunna's palm. The windmill blade cuts one beat from her long-A hook "
    "and leaves the scorecard hanging. The chosen adult counters with nine-oh-five and a second "
    "blue ball; two rolls restore the full bar. Rim click, guiro rasp, organ bubble and two cup "
    "clicks return the game to their hands."
)
p2["phase1_slot"]["verse_structure"] = (
    "Hook at zero → scorecard Round 1 → wager refrain → Round 2 counter-time → blade-cut one-bar "
    "interruption → two-roll answer hook restores meter → exact final title → bass/organ receipt."
)
p2["forced_differentiation"] = [
    "only competitive scorecard architecture",
    "only 105 BPM dembow bed",
    "only meter removed by a physical blade then restored by two rolls",
    "answer is a counter-wager, not acceptance of an appointment",
]

pairs[2]["medium"] = (
    "Sixties soul-pop at 109 BPM with walking bass, tight kit, tambourine, piano, three-voice "
    "answer stack, short horn punctuation and Sunna: late-20s-to-early-30s adult female, light "
    "Scandinavian English accent, bright authoritative mezzo. A zipper-resolve clean-guitar "
    "phrase brackets the vocal. Hard power-pop guitar enters only when the zipper closes and Sunna turns."
)
p3 = pairs[2]
p3["concept"] = (
    "At a tailor's pickup counter, Sunna names the impossible yellow jumpsuit hers and directs one "
    "chosen adult through a stop/start fastening game. Each zipper tooth earns one antiphonal line; "
    "Sunna says stop before the turn, chooses when to resume, and names Saturday only in the coda "
    "after the zipper closes."
)
p3["singer"].update(
    physical_situation="She stands half-fastened in a yellow jumpsuit at the tailor's counter, one hand on the zipper pull and one hand raised to stop the chosen adult.",
    unsaid_ache="She wants help without surrendering pace, and she wants her stop and her yes to sound equally flirtatious.",
)
p3["surface"].update(
    dominant_body_image="Her raised palm stops the zipper; the same hand lowers, turns, and taps the closed seam when she chooses to continue.",
    plain_language_ache="Follow my stop, follow my yes, then meet me Saturday.",
)
p3["golden_seed_alloy"].update(
    mythic_material_escalation="One zipper tooth becomes a bright ladder of Saturday lights, then returns to the metal pull warm under Sunna's thumb.",
    exact_detail="Saturday at seven-forty appears only in the coda after the fastening is complete.",
    controlled_fracture="The groove stops on Sunna's raised palm, resumes on her lowered hand, and the hard-guitar seam arrives only after closure.",
    routing_destination="Zipper teeth to antiphonal lines; bright ladder image must be sung once; stop/resume to form; time coordinate to coda only.",
)
p3["phase1_slot"]["verse_structure"] = (
    "Zipper-riff pickup → alternating tooth-lines → Sunna stop cue at mid-verse → two-beat held answer → "
    "resume antiphony → closure/guitar seam → one compact Saturday coda → zipper-riff receipt."
)
p3["forced_differentiation"] = [
    "only stop/start consent antiphony",
    "only future coordinate withheld until coda",
    "only groove stop commanded by the singer's raised palm",
    "only hard-guitar seam waits for physical closure",
]

p4 = pairs[3]
p4["seed_title"] = "ONE LIME, RED CHAIR"
p4["concept"] = (
    "On a marina dock before the next low tide, Sunna lays out one lime and one red folding chair. "
    "The chosen adult asks Salt too?; Sunna answers yes and hands over the shaker. A cumulative "
    "ingredient call changes vocal owner before the exact title returns, making appetite—not an "
    "appointment—the social event."
)
p4["medium"] = (
    "Bass-led rocksteady at 102 BPM with walking electric bass, cross-stick kit, offbeat clean "
    "guitar, organ bubble, sparse brass and Sunna: late-20s-to-early-30s adult female, light "
    "Scandinavian English accent, intimate full-chest mezzo. A five-note bass pickup and two guitar "
    "notes bracket the vocal. Dembow hand percussion enters only when the salt shaker changes hands."
)
p4["singer"].update(
    physical_situation="She stands on a marina dock with one lime, one red folding chair and an unopened salt shaker while the next low tide darkens the boards.",
    unsaid_ache="She wants appetite and need to be playful shared labor, not proof of dependence.",
)
p4["surface"].update(
    dominant_body_image="She squeezes lime over her own thumb, passes the salt, and opens the red chair toward the chosen adult.",
    plain_language_ache="Bring your appetite; help me season this little night.",
    candidate_hooks=["One Lime, Red Chair", "Salt Too? Yes, Right There", "Next Low Tide", "Your Hand on the Shaker"],
    panel_lean_hook="One Lime, Red Chair",
)
p4["golden_seed_alloy"].update(
    mythic_material_escalation="One lime and one salt shaker enlarge into the whole green-and-white tide line, then return as sting on Sunna's thumb and salt passed hand to hand.",
    exact_detail="Next low tide is the sole future coordinate; no clock time appears.",
    controlled_fracture="Salt too? steals the lead for one call; Sunna's yes hands it back while the percussion enters.",
    routing_destination="Lime/salt/chair to cumulative hook; tide-scale image must be sung once; shaker handoff to seam; symbolism to ghost bank.",
)
p4["triple_arch_ritual_paragraph"] = (
    "One lime sits on a red chair. Salt too? arrives from the chosen adult and the shaker crosses "
    "hands. For one sung image the dock becomes a green-and-white tide line, then citrus stings "
    "Sunna's thumb and the scale returns to appetite. Bass pickup, rind squeak, cross-stick gap and "
    "hand percussion leave the chair open."
)
p4["phase1_slot"]["verse_structure"] = (
    "Exposed bass hook → ingredient call 1 → answer line → cumulative lime/chair refrain → ingredient "
    "call 2 led by the other adult → salt handoff seam → exact final title → bass/guitar receipt."
)
p4["forced_differentiation"] = [
    "only appetite/seasoning emotional core",
    "only future coordinate is a tide, not a clock",
    "only cumulative call-response with vocal ownership change",
    "only salt-shaker handoff triggers the seam",
]

p5 = pairs[4]
p5["concept"] = (
    "At a four-shot photo booth, Sunna keeps the glossy strip the machine marks wrong. She tears the "
    "four frames apart and lets the chosen adult pick one; the selected frame determines the next "
    "stanza order. Tomorrow appears only as the printed footer. PRETTY BUSY TOMORROW changes from booth "
    "instruction to the life she schedules by choosing what gets kept."
)
p5["medium"] = (
    "Baroque sunshine-pop cut-up rondeau at 121 BPM with percussive harpsichord, melodic bass, tight "
    "kit, tambourine, pizzicato strings, close three-part answers and Sunna: late-20s-to-early-30s "
    "adult female, light Scandinavian English accent, precise playful mezzo. A five-note harpsichord "
    "cell brackets the vocal. One ska-horn answer enters when the chosen adult selects a frame; the "
    "torn strip reorders the return."
)
p5["singer"].update(
    physical_situation="She stands beside a four-shot booth with one glossy strip, tears between the frames, and fans them toward the chosen adult.",
    unsaid_ache="She wants the right to select her own visible self and tomorrow without arguing with the machine that classified the strip.",
)
p5["surface"].update(
    dominant_body_image="Four small faces fan in her hand; one chosen frame crosses to the other adult and the remaining three change the song order.",
    plain_language_ache="Pick the picture you want to keep; I will pick what happens next.",
)
p5["golden_seed_alloy"].update(
    mythic_material_escalation="One booth flash becomes four tiny bright windows and a clockwork court of neutral faces, then returns to torn paper warm between two hands.",
    exact_detail="Tomorrow is printed once in the booth footer; no clock coordinate is sung.",
    controlled_fracture="The photo strip tears into four; the chosen frame dictates a one-time stanza reorder and the horn answers selection.",
    routing_destination="Four frames to lyric and rondeau order; tiny flashbulb court must be sung once; chosen-frame handoff to seam; classification explanation to sidecar only.",
)
p5["triple_arch_ritual_paragraph"] = (
    "One glossy strip slides from the booth. Four frames become four tiny windows while the booth "
    "behaves like a clockwork court demanding neutral faces. Sunna tears between them, lets one warm "
    "hand choose a frame, and returns to the paper held between two people. Harpsichord ticks, paper "
    "tears and one horn answer reorder the rondeau."
)
p5["phase1_slot"]["verse_structure"] = (
    "Harpsichord cell → four-frame refrain → Frame 1/2 couplets → tear SFX → chosen-frame answer "
    "determines Frame 3/4 order → tiny-court bridge → transformed title return → harpsichord receipt."
)
p5["forced_differentiation"] = [
    "only cut-up photo-strip rondeau",
    "only chosen object determines section order",
    "only NEWS lane whose future is printed rather than spoken as a time",
    "only tiny clockwork court must enter sung lyric and return to hands",
]

pairs[5]["medium"] = (
    "Glam-funk power pop at 124 BPM with clavinet, strutting melodic bass, stomp-clap kit, clean "
    "guitar doubles, tambourine, stacked title vocals and Sunna: late-20s-to-early-30s adult female, "
    "light Scandinavian English accent, commanding chest-led mezzo. A syncopated clavinet/guitar "
    "cell brackets the voice. Clipped two-tone guitar takes over only when the pen touches the chosen pink page."
)

PATH.write_text(json.dumps(pairs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"wrote {PATH} with {len(pairs)} differentiated pair contracts")
