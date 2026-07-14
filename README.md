<div align="center">

# SAM-Audio

[![arXiv](https://img.shields.io/badge/arXiv-2512.18099-b31b1b.svg)](https://arxiv.org/abs/2512.18099)
![CI](https://github.com/facebookresearch/sam-audio/actions/workflows/ci.yaml/badge.svg)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-Collection-orange?logo=huggingface)](https://huggingface.co/collections/facebook/sam-audio)

![model_image](assets/sam_audio_main_model.png)

</div>

Segment Anything Model for Audio [[**Blog**](https://ai.meta.com/blog/sam-audio/)] [[**Paper**](https://ai.meta.com/research/publications/sam-audio-segment-anything-in-audio/)] [[**Demo**](https://aidemos.meta.com/segment-anything/editor/segment-audio)]

SAM-Audio is a foundation model for isolating any sound in audio using text, visual, or temporal prompts. It can separate specific sounds from complex audio mixtures based on natural language descriptions, visual cues from video, or time spans.

SAM-Audio and the Judge model crucially rely on [Perception-Encoder Audio-Visual (PE-AV)](https://huggingface.co/facebook/pe-av-large), which you can read more about [here](https://ai.meta.com/research/publications/pushing-the-frontier-of-audiovisual-perception-with-large-scale-multimodal-correspondence-learning/)

## Setup

**Requirements:**
- Python >= 3.11
- CUDA-compatible GPU (recommended)

Install dependencies:

```bash
pip install .
```

## API server

Install the API dependencies and point the service at a downloaded checkpoint:

```bash
pip install '.[api]'
SAM_AUDIO_MODEL=/models/sam-audio-small-tv sam-audio-api
```

The Docker image contains the application and its dependencies, but no model weights.
Mount a checkpoint directory at runtime:

```bash
docker build -t sam-audio-api .
docker run --gpus all --rm -p 8000:8000 \
  -v /path/to/models:/models:ro \
  sam-audio-api
```

The API runs the configured two-stage cascade (`music soundtrack`, then
`human voices`) through one continuous batcher. Submit an audio file; the response
ZIP contains `stage1_music.wav`, `stage1_residual.wav`, `stage2_voice.wav`,
`stage2_residual.wav`, and `metadata.json`. The metadata includes
`cascade_order` and a canonical stem-to-file mapping so clients do not need to
infer content from stage numbers:

```bash
curl -f http://localhost:8000/v1/separate \
  -F audio=@mixture.wav \
  -o separation.zip
```

The optional `order` form field accepts `music_first` (default) or
`voice_first`. The durable pipeline starts music-first and automatically retries
voice-first when the first result is a failure, then retains the route with the
stronger final/stage status and Judge evidence:

```bash
curl -f http://localhost:8000/v1/separate \
  -F audio=@mixture.wav \
  -F order=voice_first \
  -o separation.zip
```

The optional `targets` form field accepts `music`, `voice`, or
`music,voice` (default). A single target runs only one generation stage and the
ZIP contains that target plus its SFX residual. The durable pipeline sets this
field from its Audio Flamingo source-scene presence preflight:

```bash
curl -f http://localhost:8000/v1/separate \
  -F audio=@mixture.wav \
  -F targets=voice \
  -o voice-and-residual.zip
```

The durable pipeline also sums the stored stereo-mapped stems after separation,
reassembles the chunk joins into one joined stereo WAV per original record, and
records a 0–100 sample-aligned reconstruction similarity score plus correlation,
level, error, SNR, and per-channel metrics. These artifacts, chunk diagnostics,
and the dataset-wide source-score distribution are available in the **Split
data** dashboard. Each original also reports mono/stereo layout, codec/container,
sample rate, bit depth, bitrate, and a lossless/lossy quality tier. The direct
model ZIP API remains unchanged.

A separate resumable sampler can build private, mix-biased datasets directly
from general YouTube search (not AudioSet). It downloads only a short source
section, produces an exact ten-second PCM16/48 kHz stereo WAV, and rejects low
source bitrate/sample rate, duplicated mono, quiet or silent audio, clipping,
and duration errors:

```bash
cd pipeline
uv run sam-pipeline-youtube-random \
  --output ~/Downloads/youtube-random-1000-20260714 \
  --total 1000 --seed 20260714
```

Search metadata is only a candidate generator. Use the M2D validation command
described in `docs/pipeline/OPERATIONS.md` before treating a clip as spoken
dialogue over an active background. The gate explicitly rejects singing,
rapping, choir, and other vocal music.

Generation, prompt, batching, and TF32 policy are configured with the
`SAM_AUDIO_*` environment variables demonstrated in
[`deploy/start-sam-audio-api.sh`](deploy/start-sam-audio-api.sh). A supplied
`description` form field is accepted for compatibility but ignored in cascade mode.

`metadata.json` records the per-stage Judge `overall`, `recall`, `precision`, and
`faithfulness` values, CLAP scores, ensemble candidate scores, selected candidate,
runner-up margin, adaptive expansion, and inference settings. Judge values are
continuous quality estimates where higher is better, not calibrated probabilities.
The candidate margin measures preference over the runner-up, not correctness.

The production defaults generate candidates in rounds of four (4, then 8, then
12). Another round is generated when the ensemble margin is below `0.05` or the
stage has not reached its success threshold. Voice ranking weights Judge overall
and precision equally; music ranking uses 25% overall and 75% precision to more
strongly penalize SFX spillover.

The top-level result is `verification_status`: `success`, `uncertain`, or
`failure`. Voice thresholds default to 4.5/4.3 (success/failure), and music
thresholds to 4.4/4.1. Scores between those boundaries are `uncertain`; an
otherwise successful score with unresolved candidate margin after 12 candidates
is also `uncertain`. A failed or uncertain first stage never cancels the second
stage. Each stage keeps its numeric Judge evidence under `verification`, but there
is no top-level 1–5 score.

```bash
unzip separation.zip -d separation
jq . separation/metadata.json
```

Each response includes a standard `Server-Timing` header plus
`X-SAM-Audio-Upload-Ms`, `X-SAM-Audio-Inference-Ms`,
`X-SAM-Audio-Package-Ms`, and `X-SAM-Audio-Server-Ms` headers. Use curl's
`time_starttransfer` and `time_total` values to measure client download time.
The same headers also expose per-stage raw queue wait, preprocessing, GPU queue
wait, preparation, generation, audio decoding, CLAP, Judge, ensemble combining,
total scoring, selection, postprocessing, and stage-total durations. For example,
`X-SAM-Audio-Stage2-Generation-Ms`, `X-SAM-Audio-Stage2-CLAP-Ms`, and
`X-SAM-Audio-Stage2-Judge-Ms`. The structured values and per-round generation
durations are also stored under `inference_timings_ms` in `metadata.json`.

For a persistent Linux deployment, customize and install the example
[`deploy/sam-audio-api.service`](deploy/sam-audio-api.service) systemd unit. It
binds to localhost by default; use an SSH tunnel to access port 8000 securely.

## Durable AWS pipeline

The model-free [`pipeline`](pipeline) package adds persistent datasets, direct
S3 multi-file uploads, 30-second chunks with 5-second overlap, sound gating,
single-consumer SAM and Audio Flamingo queues, DynamoDB reconciliation, an
operations dashboard, and keyboard-first review. Model checkpoints are runtime
mounts/downloads on the GPU host and are never part of either Docker build
context.

The implementation contract, live deployment evidence, and operator commands
are maintained in:

- [`docs/pipeline/ARCHITECTURE.md`](docs/pipeline/ARCHITECTURE.md)
- [`docs/pipeline/PROGRESS.md`](docs/pipeline/PROGRESS.md)
- [`docs/pipeline/OPERATIONS.md`](docs/pipeline/OPERATIONS.md)

## Usage

⚠️ Before using SAM Audio, please request access to the checkpoints on the SAM Audio
Hugging Face [repo](https://huggingface.co/facebook/sam-audio-large). Once accepted, you
need to be authenticated to download the checkpoints. You can do this by running
the following [steps](https://huggingface.co/docs/huggingface_hub/en/quick-start#authentication)
(e.g. `hf auth login` after generating an access token.)

### Basic Text Prompting

```python
from sam_audio import SAMAudio, SAMAudioProcessor
import torchaudio
import torch

model = SAMAudio.from_pretrained("facebook/sam-audio-large")
processor = SAMAudioProcessor.from_pretrained("facebook/sam-audio-large")
model = model.eval().cuda()

file = "<audio file>" # audio file path or torch tensor
description = "<description>"

batch = processor(
    audios=[file],
    descriptions=[description],
).to("cuda")

with torch.inference_mode():
    # NOTE: `predict_spans` and `reranking_candidates` have a large impact on performance.
    # Setting `predict_span=True` and `reranking_candidates=8` will give you better results at the cost of
    # latency and memory. See the "Span Prediction" section below for more details
   result = model.separate(batch, predict_spans=False, reranking_candidates=1)

# Save separated audio
sample_rate = processor.audio_sampling_rate
torchaudio.save("target.wav", result.target.cpu(), sample_rate)      # The isolated sound
torchaudio.save("residual.wav", result.residual.cpu(), sample_rate)  # Everything else
```

### Prompting Methods

SAM-Audio supports three types of prompts:

1. **Text Prompting**: Describe the sound you want to isolate using natural language. To match training, please use lowercase noun-phrase/verb-phrase (NP/VP) format for text (for example instead of "Thunder can be heard in the background" use "thunder").
   ```python
   processor(audios=[audio], descriptions=["man speaking"])
   ```

2. **Visual Prompting**: Use video frames and masks to isolate sounds associated with visual objects
   ```python
   processor(audios=[video], descriptions=[""], masked_videos=processor.mask_videos([frames], [mask]))
   ```

3. **Span Prompting**: Specify time ranges where the target sound occurs
   ```python
   processor(audios=[audio], descriptions=["car honking"], anchors=[[["+", 6.3, 7.0]]])
   ```

See the [examples](examples) directory for more detailed examples

### Span Prediction (Optional for Text Prompting)

We also provide support for automatically predicting the spans based on the text description, which is especially helpful for separating non-ambience sound events.  You can enable this by adding `predict_spans=True` in your call to `separate`

```python
with torch.inference_mode()
   outputs = model.separate(batch, predict_spans=True)

# To further improve performance (at the expense of latency), you can add candidate re-ranking
with torch.inference_mode():
   outputs = model.separate(batch, predict_spans=True, reranking_candidates=8)
```

### Re-Ranking

We provide the following models to assess the quality of the separated audio:

- [CLAP](https://github.com/LAION-AI/CLAP): measures the similarity between the target audio and text description
- [Judge](https://huggingface.co/facebook/sam-audio-judge): measures the overall separation quality across 3 axes: precision, recall, and faithfulness (see the [model card](https://huggingface.co/facebook/sam-audio-judge#output-format) for more details)
- [ImageBind](https://github.com/facebookresearch/ImageBind): for visual prompting, we measure the imagebind embedding similarity between the separated audio and the masked input video

We provide support for generating multiple candidates (by setting `reranking_candidates=<k>` in your call to `separate`), which will generate `k` audios, and choose the best one based on the ranking models mentioned above

# Models

Below is a table of each of the models we released along with their overall subjective evaluation scores

| Model    | General SFX | Speech | Speaker | Music | Instr(wild) | Instr(pro) |
|----------|-------------|--------|---------|-------|-------------|------------|
| [`sam-audio-small`](https://huggingface.co/facebook/sam-audio-small) | 3.62        | 3.99   | 3.12    | 4.11  | 3.56        | 4.24       |
| [`sam-audio-base`](https://huggingface.co/facebook/sam-audio-base)   | 3.28        | 4.25   | 3.57    | 3.87  | 3.66        | 4.27       |
| [`sam-audio-large`](https://huggingface.co/facebook/sam-audio-large) | 3.50        | 4.03   | 3.60    | 4.22  | 3.66        | 4.49       |

We additional release another variant (in each size) that is better specifically on correctness of target sound as well as visual prompting:
- [`sam-audio-small-tv`](https://huggingface.co/facebook/sam-audio-small-tv)
- [`sam-audio-base-tv`](https://huggingface.co/facebook/sam-audio-base-tv)
- [`sam-audio-large-tv`](https://huggingface.co/facebook/sam-audio-large-tv)

## Evaluation

See the [eval](eval) directory for instructions and scripts to reproduce results from the paper

## Contributing

See [contributing](CONTRIBUTING.md) and [code of conduct](CODE_OF_CONDUCT.md) for more information.

## License

This project is licensed under the SAM License - see the [LICENSE](LICENSE) file for details.

## Citing SAM Audio

If you use SAM Audio in your research, please use the following BibTex entry:

```bibtex
@article{shi2025samaudio,
    title={SAM Audio: Segment Anything in Audio},
    author={Bowen Shi and Andros Tjandra and John Hoffman and Helin Wang and Yi-Chiao Wu and Luya Gao and Julius Richter and Matt Le and Apoorv Vyas and Sanyuan Chen and Christoph Feichtenhofer and Piotr Doll{\'a}r and Wei-Ning Hsu and Ann Lee},
    year={2025},
    url={https://arxiv.org/abs/2512.18099}
}
```
