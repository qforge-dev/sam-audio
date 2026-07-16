import Config

config :logger, :console,
  format: "$time $metadata[$level] $message\n",
  metadata: [:stage, :request_id, :candidate_id]

config :sam_source_pipeline,
  search_adapter: SamSourcePipeline.Adapters.Dummy.Search,
  enrichment_adapter: SamSourcePipeline.Adapters.Dummy.Enrichment,
  repository_adapter: SamSourcePipeline.Adapters.Dummy.Repository,
  enrichment_workers: 2,
  demand: [
    search: [min_demand: 1, max_demand: 2],
    deduplication: [min_demand: 2, max_demand: 4],
    enrichment: [min_demand: 1, max_demand: 2],
    qualification: [min_demand: 2, max_demand: 4],
    sink: [min_demand: 2, max_demand: 4]
  ]
