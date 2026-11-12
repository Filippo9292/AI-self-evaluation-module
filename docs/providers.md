# Provider presets and qualification

Documentation review date: **2026-11-12**. API availability and limits may change. No paid/live provider was called to qualify this repository during implementation. Offline tests verify payload construction and strict answer validation; they do not establish backend image understanding or accuracy.

## Implemented transports

| Preset | Image transport | Endpoint | Verification |
| --- | --- | --- | --- |
| jev | Named `assets`, base64 RGB PNG | `/v1/systemone` | Offline contract tests; live pending |
| liquid | Top-level ordered `images` data URLs | `/decisions/v1/systemone` | Offline contract tests; live pending |
| drex | Top-level ordered `images` data URLs | `/v1/systemone` | Offline contract tests; live pending |
| perplexity | `image_url` parts in `state` | `/v1/decisions` | Experimental; live pending |
| mercury | `image_url` parts in `state` | `/v1/decisions` | Experimental; live pending |
| baseten | User-message content under `state.messages` | `/v1/systemone` | Experimental; live pending |
| celeris | Named data URLs under `x_celeris.images` | `/v1/systemone` | Experimental; six-image limit unverified |

All adapters ask the same configured question on the same ordered five-level rubric. They send one candidate plus five real references and do not use model-specific prompt optimization. Input formatting and undocumented visual preprocessing still differ, so equal numeric scales do not establish calibrated equivalence across providers. Each provider needs its own real baseline. Provider confidence is stored but is not used for stopping or compared as a universal quantity.

The package verifies a non-empty response model identifier and freezes the first resolved identifier for a run. Some services echo an alias or a stable public ID without exposing an immutable weights revision. A matching ID cannot guarantee frozen backend weights. Built-in presets prefer explicit model IDs where available; Baseten's default slug includes `latest` and must be treated accordingly.

## Primary documentation

- [TypeSafe images](https://docs.typesafe.ai/images), [Score](https://docs.typesafe.ai/primitives/score), [System One](https://docs.typesafe.ai/api/system-one): named image assets, an image-enabling header and a minimum of 280 pixels on each side. The JEV preset declares that minimum for shared candidate/reference preprocessing.
- [Liquid make decisions](https://docs.liquid.ai/api/reference/decision-models/make-decisions), [Decisions guide](https://docs.liquid.ai/api/guides/decisions): score questions and an ordered image array. Up to eight images, with body-size constraints.
- [Drex Max](https://console.nace.ai/docs/drex/guides/drex-max): `drex-max-v1.1`, up to eight images, numbered by array order.
- [Perplexity decision endpoint](https://docs.perplexity.ai/api-reference/decisions-post): score questions and base64 image parts, up to eight images per question. No web-grounding mode is requested.
- [Mercury Decide](https://docs.inceptionlabs.ai/capabilities/decisions): rubric score and image parts, up to sixteen images.
- [Baseten decision models](https://docs.baseten.co/inference/model-apis/decision-models), [System One reference](https://docs.baseten.co/reference/inference-api/system-one): conversation-state image inputs and up to sixteen images at the endpoint. Qualify the selected model slug individually.
- [Celeris Decision](https://celeris.ai/decisions): named `x_celeris.images` and System One questions. Confirm six-image support on the service before advertising it as qualified.

The repository request-size default is deliberately conservative (4,000,000 encoded bytes). Large images may need a smaller shared preprocessing size or a qualified provider-specific override. Never silently remove references to fit a limit.

## Excluded from this release

- [SubQ Snap](https://docs.subq.ai/endpoints/decisions-api/): documented four-image cap.
- [Fastino GLiDE](https://docs.fastino.ai/inference/systemone): documented four-image cap.
- [Microsoft-Decision-1](https://learn.microsoft.com/en-us/azure/foundry/foundry-models/how-to/use-foundry-models-microsoft-decision): the reviewed guide documents text/JSON; six-image evaluation is not established.
- [OpenAI Decisions](https://developers.openai.com/api/docs/guides/decisions): images and score exist, but the request/answer contract differs from the initial System One adapter family. No OpenAI adapter is claimed in this release.
- Clef: an image-capable family, but a six-image transport compatible with this implementation has not been qualified. It is not an advertised built-in preset.

## Qualification procedure

Use a dedicated output directory and a small, non-sensitive image set. Set the provider/API key, build a small baseline, and evaluate known acceptable and clearly incompatible candidates. Check that six images are accepted, labels bind to the intended pictures, probabilities/score are valid, the response model is expected, and judgments respond sensibly to the visual contrasts. Inspect `requests.jsonl` and actual usage. Include repeated identical requests in a separate qualification experiment to isolate service variability from reference variation. Pin the configuration/commit and retain evidence before changing the verification status in the README. No automated paid qualification is run by CI.
