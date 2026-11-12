# Credentials and private data

Use a private `evaluation.local.yaml` for an inline API key, or the optional provider environment variable. Neither private YAML nor `.env` files belong in a public commit. The software omits keys/authorization headers from output and redacts the configured key if it is echoed by a provider response.

Live evaluation transmits candidate and reference images to the selected provider. Use an appropriate account and data-sharing policy for your dataset. Output manifests retain local paths and hashes; raw response text may contain provider-specific information. Review outputs before sharing them.

Provider presets can change the endpoint; use trusted configuration files. No fallback endpoint or provider is selected automatically. The package does not load/execute model checkpoint contents: the integrating trainer is responsible for safe deserialization.
