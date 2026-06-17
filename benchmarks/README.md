# H100 benchmark inputs

This directory contains small benchmark manifests only. Do not commit audio files.

Use `prompts.json` as the default prompt set, and point the H100 runner at a
local directory of audio files:

```bash
scripts/run_h100_matrix.sh /path/to/audio_dir /path/to/benchmark_runs
```

The benchmark script scans common audio extensions in the audio directory and
runs every prompt in the prompt manifest against each audio file.

For an apples-to-apples single-call breakdown like:

- encoding
- span prediction
- ODE generation
- audio decoding
- reranking

use a prompt file with one prompt and one 30s audio file. The default
`prompts.json` batches several prompts, which is better for throughput testing
but produces different stage percentages.
