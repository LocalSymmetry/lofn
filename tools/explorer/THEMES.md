# Lofn Explorer Theme Record

Created: 2026-07-04

> *Panel voices are model-generated interpretive constructs, each "after" a named source figure's published work. No statement is a quotation of, or endorsement by, the named person.*

## Panel

- INTERFACE SEMANTICIST (after Susan Kare): favor icons that are mnemonic at 16px and not merely decorative.
- COLOR RELATIVITY CARTOGRAPHER (after Josef Albers): every theme needs a distinct value structure, not just a hue swap.
- TYPOGRAPHIC SYSTEMS EDITOR (after Ellen Lupton): typography changes should alter reading tempo without compromising dense operator surfaces.
- ORNAMENT HISTORIAN (after Owen Jones): decorative systems must become interface grammar: rails, gates, marks, borders.
- DATA-VIZ MINIMALIST (after Edward Tufte): do not let theme art reduce data contrast or make severity colors ambiguous.
- HYPER-SKEPTIC (after Adolf Loos): theme novelty is a failure if it becomes costume jewelry over an unchanged app.

## Panel Decisions

The theme system must change three axes together:

1. Color tokens: background, surface stack, accents, severity tones, and rail hues.
2. Typography tokens: display, body, and mono stacks, plus radius/shadow rhythm where needed.
3. Iconography tokens: brand mark and per-nav glyph set, styled through theme variables.

The panel rejected one-note palette variants. Each added theme needs a different material metaphor and a different operator mood.

## Theme Roster

| Theme | Art Asset | Typography | Iconography |
| --- | --- | --- | --- |
| Gilded Vellum | CSS fallback | literary serif + operator mono | manuscript marks, gilt thread |
| Sunset Synthwave | `web/public/theme-art/sunset-synthwave.png` | retrowave geometric | chrome horizon marks |
| 8-bit | `web/public/theme-art/pixel-arcade.png` | monospace-first | pixel blocks and status glyphs |
| Pastel Butterfly | `web/public/theme-art/pastel-butterfly.png` | airy serif + soft system UI | wing geometry and specimen pins |
| Art Nouveau Fairies | `web/public/theme-art/art-nouveau-fairies.png` | ornamental serif | vines, jewels, arch marks |
| Lunar Ink Observatory | `web/public/theme-art/lunar-ink-observatory.png` | quiet serif + observatory mono | moons, lenses, star pins |
| Solarpunk Glasshouse | `web/public/theme-art/solarpunk-glasshouse.png` | warm botanical serif | leaf veins and seed-pod gates |
| Bioluminescent Tidepool | `web/public/theme-art/bioluminescent-tidepool.png` | clear sans + liquid mono | tide marks and anemone dots |
| Gothic Botanical Lab | `web/public/theme-art/gothic-botanical-lab.png` | grave serif + surgical mono | wax seals and specimen plates |
| Bauhaus Signal Room | `web/public/theme-art/bauhaus-signal-room.png` | constructivist condensed labels | circles, squares, triangles |
| Desert Chrome Mirage | `web/public/theme-art/desert-chrome-mirage.png` | industrial sans | cairns, horizons, route marks |
| Paper Lantern Rain | `web/public/theme-art/paper-lantern-rain.png` | paper-serif headings | thread paths and lantern seals |
| Arctic Aurora Codex | `web/public/theme-art/arctic-aurora-codex.png` | crisp sans + expedition mono | field lines, prisms, beacons |

## Lofn Side-Door Prompt Method

The theme art used Lofn side-door prompt work: quick project-bound visual fragments, not a full competition run. Prompts followed the GPT-Image-2 renderer rules in `skills/image/renderer_gpt_image2_rules.md`: one shared light source, strong material physics, no labels/text, one coherent scene rather than collage, and crisp thumbnail-readable icon silhouettes.

The literal CLI `gpt-image-2` path was available, but `OPENAI_API_KEY` was not set in this shell. The actual renders were produced with the built-in image generation tool and copied into `web/public/theme-art/`, leaving the original Codex cache files in place.

## Prompt Nuclei

- Sunset Synthwave: chrome pipeline rails, black glass, low sunset horizon, magenta/cyan neon, brass-bound Lofn notebook, no text.
- 8-bit: pixel-art operations room inside a cartridge, CRT beam, chunky validation locks, phosphor green and amber status lights.
- Pastel Butterfly: conservatory drafting desk, butterfly wing-vein rails, translucent specimen pins, frosted glass and pearl enamel.
- Art Nouveau Fairies: moonlit stained-glass control arbor, vine rails, jeweled clasp gates, seven abstract fairy-like UI motifs.
- Lunar Ink Observatory: sumi ink observatory desk, star-node pipeline, eclipsed-moon gates, vellum chart and silver search lens.
- Solarpunk Glasshouse: greenhouse operations console, copper trellises, seed-pod gates, transparent plant-cell library shelves.
- Bioluminescent Tidepool: tidal lab, plankton rail trails, coral-like gates, basalt-and-glass submerged console.
- Gothic Botanical Lab: black-stone research bench, thorned iron rails, red wax seals, herbarium cards and engraved glass.
- Bauhaus Signal Room: modernist signal-control room, circles/squares/triangles, matte metal, enamel indicators.
- Desert Chrome Mirage: dusk desert worktable, heat-haze rails, mirrored cairn gates, etched topographic glass.
- Paper Lantern Rain: rain-streaked night workbench, red thread pipeline paths, washi drawers, lantern seals, lacquer trays.
- Arctic Aurora Codex: polar research desk, aurora ribbons, ice-crystal gates, frosted glass and titanium field console.

