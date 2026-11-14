# Provider contract

The packaged providers are `jev` and `mock`. Select another compatible service using `provider_file` in the main YAML. A custom preset must define `name`, `adapter: typesafe_assets`, an HTTPS `endpoint`, a pinned `model`, `api_key_env` and any required `headers`.

The adapter posts JSON with `model`, `state`, named `questions` and `assets`. Assets contain RGB PNG bytes in `data_base64`. Question-specific references appear in instructions, not shared state. A combined request can include up to twelve distinct images; each presence question sees one candidate and ten references, each quality question six images and each relative question seven. The service must support these scoped contexts.

Responses must return the pinned model name, exactly the requested question IDs, `score` answers with finite scores in [0,4], `noul` answers in [0,1], and `choice` answers containing normalized A/B probabilities. Aliases that resolve to a different model are rejected; configure the resolved pinned name.

No automatic provider fallback or conversion from free text is implemented. Models needing chat-specific image transports or different prompting are not supported by this release. The mock is a synthetic software fixture.

The JEV preset checks prepared-image dimensions before dispatch: both sides must be between 280 and 1568 pixels. The default preprocessing enlarges small images to a minimum side of 280 while preserving aspect ratio; incompatible aspect ratios or explicit sizes fail locally. These settings remain editable in preprocessing YAML, subject to the selected provider's limits.

Contracts: [System One](https://docs.typesafe.ai/api/system-one), [images and scoped assets](https://docs.typesafe.ai/images).
