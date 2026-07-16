defmodule SamSourcePipeline.Stages.Enrichment do
  @moduledoc "Concurrent adapter boundary for metadata and proxy audio analysis."

  use GenStage

  require Logger

  alias SamSourcePipeline.{AdapterCall, Domain.Candidate, Domain.Result}

  def start_link(opts) do
    GenStage.start_link(__MODULE__, opts, name: Keyword.fetch!(opts, :name))
  end

  @impl true
  def init(opts) do
    worker = Keyword.fetch!(opts, :name)
    Logger.metadata(stage: worker)

    state = %{adapter: Keyword.fetch!(opts, :adapter), worker: worker}
    producer = Keyword.fetch!(opts, :producer)
    demand = Keyword.fetch!(opts, :demand)

    {:producer_consumer, state, subscribe_to: [{producer, demand}]}
  end

  @impl true
  def handle_events(events, _from, state) do
    events = Enum.map(events, &enrich(&1, state))
    {:noreply, events, state}
  end

  defp enrich(%Result{} = result, _state), do: result

  defp enrich(%Candidate{} = candidate, state) do
    Logger.info("enriching source",
      request_id: candidate.request_id,
      candidate_id: candidate.id
    )

    case AdapterCall.run(fn -> state.adapter.enrich(candidate) end) do
      {:ok, enriched} ->
        enriched

      {:error, reason} ->
        Logger.warning("enrichment failed: #{inspect(reason)}",
          request_id: candidate.request_id,
          candidate_id: candidate.id
        )

        Result.candidate_failed(candidate, :enrichment, reason)
    end
  end
end
