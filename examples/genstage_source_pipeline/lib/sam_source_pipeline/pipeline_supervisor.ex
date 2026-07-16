defmodule SamSourcePipeline.PipelineSupervisor do
  @moduledoc "Builds the complete source-discovery demand graph."

  use Supervisor

  alias SamSourcePipeline.Adapters.Dummy.Repository, as: DummyRepository
  alias SamSourcePipeline.Stages

  def start_link(opts \\ []) do
    Supervisor.start_link(__MODULE__, opts, name: Keyword.get(opts, :name, __MODULE__))
  end

  @impl true
  def init(opts) do
    search_adapter = adapter(opts, :search_adapter)
    enrichment_adapter = adapter(opts, :enrichment_adapter)
    repository_adapter = adapter(opts, :repository_adapter, DummyRepository)
    worker_count = option(opts, :enrichment_workers, 2)
    demand = option(opts, :demand, [])

    if worker_count < 1, do: raise(ArgumentError, "enrichment_workers must be positive")

    enrichment_workers =
      Enum.map(1..worker_count, fn index ->
        Module.concat(SamSourcePipeline, "EnrichmentWorker#{index}")
      end)

    enrichment_children =
      Enum.map(enrichment_workers, fn name ->
        Supervisor.child_spec(
          {Stages.Enrichment,
           name: name,
           producer: Stages.Deduplication,
           adapter: enrichment_adapter,
           demand: demand_for(demand, :enrichment)},
          id: name
        )
      end)

    children =
      [
        repository_adapter.child_spec(name: repository_adapter),
        {SamSourcePipeline.QueryQueue, name: SamSourcePipeline.QueryQueue},
        {Stages.Search,
         name: Stages.Search,
         producer: SamSourcePipeline.QueryQueue,
         adapter: search_adapter,
         demand: demand_for(demand, :search)},
        {Stages.Deduplication,
         name: Stages.Deduplication,
         producer: Stages.Search,
         demand: demand_for(demand, :deduplication)}
      ] ++
        enrichment_children ++
        [
          {Stages.Qualification,
           name: Stages.Qualification,
           producers: enrichment_workers,
           demand: demand_for(demand, :qualification)},
          {Stages.Sink,
           name: Stages.Sink,
           producer: Stages.Qualification,
           repository: repository_adapter,
           repository_server: repository_adapter,
           demand: demand_for(demand, :sink)}
        ]

    Supervisor.init(children, strategy: :rest_for_one)
  end

  defp adapter(opts, key, fallback \\ nil) do
    Keyword.get(opts, key) ||
      Application.get_env(:sam_source_pipeline, key, fallback) ||
      raise ArgumentError, "missing #{key}"
  end

  defp option(opts, key, fallback) do
    Keyword.get(opts, key, Application.get_env(:sam_source_pipeline, key, fallback))
  end

  defp demand_for(demand, stage) do
    Keyword.get(demand, stage, min_demand: 1, max_demand: 4)
  end
end
