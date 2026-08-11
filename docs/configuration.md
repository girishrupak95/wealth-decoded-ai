# Configuration

Wealth Decoded separates committed application settings from local credentials:

- `config/settings.toml` is committed and contains non-secret configuration.
- `.secrets.env` is local, ignored by Git, and contains credentials only.
- Explicit process environment variables override both files where a settings model supports the
  corresponding variable.
- Constructor arguments used for dependency injection and tests take highest precedence.

Never commit `.secrets.env`, paste its values into documentation, or put credentials in
`config/settings.toml`.

## Local secret setup

Create `.secrets.env` in the repository root only when a provider is needed:

```dotenv
WEALTH_OPENAI_API_KEY=<your-key>
ELEVENLABS_API_KEY=<your-key>
```

Required secret variable names:

| Variable | Required when |
| --- | --- |
| `WEALTH_OPENAI_API_KEY` | Calling OpenAI-backed LLM or image workflows |
| `ELEVENLABS_API_KEY` | Generating or verifying ElevenLabs voiceover |
| `WEALTH_DATABASE_URL` | Replacing the legitimate local-development database default with a credential-bearing DSN |

Missing `.secrets.env` does not block local model validation, dry runs, tests, deterministic chart
planning, typography, or other provider-free workflows. Provider construction reports the missing
credential only when that provider is requested.

## Committed settings

Edit `config/settings.toml` for non-secret configuration. Its tables cover:

- `[application]`: service identity, bind settings, logging, database pool sizing, and request IDs.
- `[openai]`: chat model, temperature, and global output-token default.
- `[storyboard]`: StoryboardAgent output-token ceiling.
- `[visual_assets]`: live-generation control, request cap, model, quality, and failure policy.
- `[voiceover]`: non-secret voice ID, model, output format, and voice controls.
- `[render]`: executable names, timeouts, and graceful termination.
- `[development]`: local diagnostic behavior that is safe to commit disabled.

Paths or settings that differ on one machine can be overridden with the existing process variable
name, for example:

```bash
FFMPEG_EXECUTABLE=/custom/bin/ffmpeg uv run python apps/api/scripts/run_demo_render.py --dry-run
```

Do not put API keys, access tokens, passwords, or credential-bearing database URLs in TOML.

## Precedence

From lowest to highest priority:

1. Legitimate model defaults.
2. `config/settings.toml`.
3. Credentials loaded from `.secrets.env`.
4. Explicitly exported process environment variables.
5. Explicit constructor values used by dependency injection or tests.

`config/settings.toml` is required. File paths are resolved from the repository root rather than
the current terminal directory. `.secrets.env` is optional.

## Provider examples

For OpenAI, place only `WEALTH_OPENAI_API_KEY` in `.secrets.env`; select chat and image models in
TOML. For ElevenLabs, place only `ELEVENLABS_API_KEY` in `.secrets.env`; select the voice, model,
format, and voice controls in TOML.

Environment overrides retain their established names, including `WEALTH_OPENAI_MODEL`,
`VISUAL_ASSET_IMAGE_MODEL`, `VISUAL_ASSET_IMAGE_QUALITY`, `ELEVENLABS_VOICE_ID`, and
`RENDER_TIMEOUT_SECONDS`.
