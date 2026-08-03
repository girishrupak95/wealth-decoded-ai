# Pipeline

## Content pipeline

```text
TopicCandidate → VideoConcept → ResearchPackage → VideoScript → ScriptReview
```

Topic, concept, research, script, and review services persist their respective editorial artifacts
under `generated/topics`, `generated/concepts`, `generated/research`, `generated/scripts`, and
`generated/reviews` where that service supports persistence. Each agent response is parsed and
validated with Pydantic. Research currently comes from the LLM output; it is not web retrieval.

`ScriptReview.approved` is the human/editorial gate. A rejected review stops downstream
storyboard, voiceover, and visual work.

## Voiceover branch

```text
Approved ScriptReview + VideoScript
  → VoiceoverManifest → audio segments → combined narration
```

The ElevenLabs provider may make paid external calls. Audio processing uses local ffmpeg/ffprobe.
Voiceover files and manifests are persisted under `generated/voiceovers`.

## Visual branch

```text
Approved ScriptReview + VideoConcept + VideoScript
  → Storyboard → VisualAssetGenerationService → VisualAssetPersistence
  → VisualAssetManifest + generated or instructed assets
```

Storyboard timing, summaries, scene coverage, warnings, and asset manifest totals are determined
or normalized deterministically. Visual packages are persisted beneath
`generated/visual-assets/YYYY-MM-DD/<safe-title>/`.

The default visual mode is manifest-only:

- AI-image scenes become pending instructions; no image provider call occurs.
- Typography scenes render locally into PNG bytes and can be persisted.
- Stock, chart, screenshot, motion-graphic, screen-recording, and unconfigured-video scenes
  remain traceable manifest instructions.
- No paid image call occurs.

Setting `VISUAL_ASSET_LIVE_GENERATION=true` enables bounded OpenAI image calls; this is the only
visual paid-provider path currently implemented. No live video provider, stock download,
automated screenshot capture, or final video assembly exists.
