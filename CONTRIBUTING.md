# Contributing

Contributions that improve fidelity, add benchmark coverage, or independently verify a result are welcome.

1. Keep each scientific change isolated in its own branch.
2. Commit configuration changes with the code they evaluate.
3. Report the immutable commit and complete terminal metrics for every run.
4. Do not overwrite a completed experiment branch; create a child branch for a new hypothesis.
5. Add new evidence to `results.csv` and update the self-contained notebook data in the same commit.
6. Document hardware, seeds, model revisions, and any deviation from the source paper.

Please avoid committing model weights, private credentials, raw access tokens, or large generated outputs.
