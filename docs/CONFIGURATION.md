# Configuration

Settings load `.env` when present. Never commit it; `.env.example` contains safe templates.

## Application and database

| Variable | Purpose | Required | Default | Secret | Used by |
| --- | --- | --- | --- | --- | --- |
| `WEALTH_APP_NAME` | Service name | No | `wealth-decoded-ai` | No | FastAPI settings |
| `WEALTH_ENVIRONMENT` | Runtime label | No | `local` | No | FastAPI settings |
| `WEALTH_HOST` / `WEALTH_PORT` | Bind address and port | No | `0.0.0.0` / `8000` | No | API server |
| `WEALTH_LOG_LEVEL` | Application log level | No | `INFO` | No | Logging |
| `WEALTH_DATABASE_URL` | Async PostgreSQL DSN | No | local PostgreSQL DSN | Contains credentials | Database session |
| `WEALTH_DATABASE_POOL_SIZE` / `WEALTH_DATABASE_MAX_OVERFLOW` | Pool limits | No | `5` / `10` | No | Database session |
| `WEALTH_REQUEST_ID_HEADER` | Correlation header | No | `X-Request-ID` | No | Request middleware |

## OpenAI / LLM

| Variable | Purpose | Required | Default | Secret | Used by |
| --- | --- | --- | --- | --- | --- |
| `WEALTH_OPENAI_API_KEY` | LLM and live-image API credential | Yes for live pipeline execution | none | Yes | `OpenAIClient`, live images |
| `WEALTH_OPENAI_MODEL` | LLM model | Yes | none | No | LLM agents |
| `WEALTH_OPENAI_TEMPERATURE` | LLM sampling temperature | Yes | none | No | LLM agents |
| `WEALTH_OPENAI_MAX_TOKENS` | LLM response limit | Yes | none | No | LLM agents |

## ElevenLabs

| Variable | Purpose | Required | Default | Secret | Used by |
| --- | --- | --- | --- | --- | --- |
| `ELEVENLABS_API_KEY` | TTS credential | Yes for voiceover CLI | none | Yes | ElevenLabs provider |
| `ELEVENLABS_VOICE_ID` | Selected voice | Yes for voiceover CLI | none | No | Voiceover service |
| `ELEVENLABS_MODEL_ID` | TTS model | No | `eleven_multilingual_v2` | No | Voiceover service |
| `ELEVENLABS_OUTPUT_FORMAT` | Audio output | No | `mp3_44100_128` | No | Voiceover service |
| `ELEVENLABS_STABILITY` / `ELEVENLABS_SIMILARITY_BOOST` / `ELEVENLABS_STYLE` | Voice settings | No | `0.5` / `0.75` / `0.0` | No | Voiceover service |
| `ELEVENLABS_USE_SPEAKER_BOOST` | Voice enhancement toggle | No | `true` | No | Voiceover service |

## Visual assets

| Variable | Purpose | Required | Default | Secret | Used by |
| --- | --- | --- | --- | --- | --- |
| `VISUAL_ASSET_LIVE_GENERATION` | Enable paid image calls | No | `false` | No | Visual CLI |
| `VISUAL_ASSET_MAX_LIVE_IMAGES` | Maximum live images per run | No | `5` | No | Visual service |
| `VISUAL_ASSET_FAIL_FAST` | Stop on first visual failure | No | `false` | No | Visual service |
| `VISUAL_ASSET_IMAGE_MODEL` | OpenAI image model | No | `gpt-image-1` | No | Visual CLI |
| `VISUAL_ASSET_IMAGE_QUALITY` | Optional image quality | No | unset | No | Visual CLI |

## Development and testing

No testing-only environment variables are currently read by the repository. Tests use mocks,
temporary paths, and injected values instead.

## Template audit

`.env.example` now includes every setting-model variable. Its database DSN targets the Docker
service (`db`), while the code default targets local development (`localhost`); choose the value
appropriate for the runtime environment. There are no duplicated template keys.
