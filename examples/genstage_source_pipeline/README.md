# Demand-driven source discovery with GenStage

This is a standalone Mix application that models the repository's source
acquisition workflow. The integrations are deterministic dummy adapters, while
the process topology, demand propagation, supervision, error outcomes, adapter
contracts, and tests are real.

## Workflow

```text
Events ──────────────────────────────────────────────────────────────────────▶

Search API
   │
   ▼
QueryQueue ──▶ Search ──▶ Deduplication ──┬─▶ EnrichmentWorker1 ─┐
                                         └─▶ EnrichmentWorker2 ─┤
                                                               ▼
                                                      Qualification ──▶ Sink
                                                                          │
                                                                          ▼
Demand ◀────────────────────────────────────────────────────────── Repository
```

The sink initiates demand. Every consumer asks its producer for a bounded
amount of work, so the in-memory query queue releases commands only when the
downstream workflow has capacity. Two enrichment stages subscribe to the same
deduplication stage and therefore form a demand-balanced worker pool.

The stages are:

1. `QueryQueue` accepts search commands and implements an explicit FIFO plus
   outstanding-demand counter.
2. `Stages.Search` calls the configured search adapter and expands one command
   into source candidates.
3. `Stages.Deduplication` owns the global provider/external-ID set. Duplicates
   become persisted rejection results rather than silently disappearing.
4. `Stages.Enrichment` calls a metadata/proxy-analysis adapter. Multiple named
   processes perform this work concurrently.
5. `Stages.Qualification` applies a pure domain policy to duration, foreground
   speech, music, and sound-effect evidence.
6. `Stages.Sink` persists all accepted, rejected, and failed terminal outcomes.

Search and enrichment adapter exceptions are converted into explicit failure
results, allowing unrelated events to keep moving through the pipeline.

## Run it

```bash
cd examples/genstage_source_pipeline
mix deps.get
mix run demo/run.exs
mix test
```

## Replace the dummy boundaries

Configuration lives in `config/config.exs`. Production implementations only
need to implement these behaviours:

- `SamSourcePipeline.Adapters.Search` — provider/catalog search.
- `SamSourcePipeline.Adapters.Enrichment` — metadata lookup and proxy audio
  analysis.
- `SamSourcePipeline.Adapters.Repository` — terminal result persistence.

The current repository adapter is an `Agent`, and the query queue is in memory.
Those choices make the example runnable but not durable. A production version
should put the accepted command in a durable store before returning from
`SamSourcePipeline.search/2`, make adapter operations idempotent, and persist
deduplication keys. GenStage supplies bounded demand and backpressure; it does
not itself provide durable delivery or exactly-once processing.

`min_demand` and `max_demand` for every subscription boundary are independently
configurable. Raising enrichment worker count increases adapter concurrency;
raising demand increases the bounded amount of buffered/in-flight work.
