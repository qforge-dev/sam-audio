defmodule SamSourcePipeline.Stages.Search do
  @moduledoc "Expands each search command into provider candidates."

  use GenStage

  require Logger

  alias SamSourcePipeline.{AdapterCall, Domain.Result}

  def start_link(opts) do
    GenStage.start_link(__MODULE__, opts, name: Keyword.fetch!(opts, :name))
  end

  @impl true
  def init(opts) do
    Logger.metadata(stage: :search)

    state = %{adapter: Keyword.fetch!(opts, :adapter)}
    producer = Keyword.fetch!(opts, :producer)
    demand = Keyword.fetch!(opts, :demand)

    {:producer_consumer, state, subscribe_to: [{producer, demand}]}
  end

  @impl true
  def handle_events(requests, _from, state) do
    events = Enum.flat_map(requests, &search(&1, state.adapter))
    {:noreply, events, state}
  end

  defp search(request, adapter) do
    Logger.info("searching provider adapters", request_id: request.id)

    case AdapterCall.run(fn -> adapter.search(request) end) do
      {:ok, candidates} when is_list(candidates) ->
        Logger.info("search produced #{length(candidates)} candidates", request_id: request.id)
        candidates

      {:ok, invalid} ->
        [Result.search_failed(request, {:invalid_candidates, invalid})]

      {:error, reason} ->
        Logger.warning("search failed: #{inspect(reason)}", request_id: request.id)
        [Result.search_failed(request, reason)]
    end
  end
end
