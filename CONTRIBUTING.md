# Contributing

Install with `python -m pip install -e .` and run `python -m unittest discover -s tests -v`. Tests and CI must remain offline and must not require private images or API credentials.

Changes to policy semantics need focused state-machine tests and documentation. Changes to scoring, preprocessing, reference sampling or request rendering require a new reproducibility fingerprint/schema version when they alter existing run meaning. Preserve raw response evidence and distinguish image count from visit/request count.

Provider additions must document six-image support, the transport and probability-weighted score contract, and their qualification status. Do not advertise paid/live verification based solely on a mocked response. Never put keys or dataset/checkpoint files in fixtures.

When reporting an issue, include the package version, operating system, Python version, provider/model ID, redacted configuration and minimal reproducible error. Review paths and provider responses before attaching them. Do not attach your private YAML containing a key.
