defmodule SamSourcePipeline.Stages.Deduplication do
  @moduledoc "Globally deduplicates provider sources before expensive enrichment."

  use GenStage

  alias SamSourcePipeline.Domain.{Candidate, Result}

  def start_link(opts) do
    GenStage.start_link(__MODULE__, opts, name: Keyword.fetch!(opts, :name))
  end

  @impl true
  def init(opts) do
    producer = Keyword.fetch!(opts, :producer)
    demand = Keyword.fetch!(opts, :demand)

    {:producer_consumer, MapSet.new(), subscribe_to: [{producer, demand}]}
  end

  @impl true
  def handle_events(events, _from, seen) do
    {events, seen} = Enum.map_reduce(events, seen, &deduplicate/2)
    {:noreply, events, seen}
  end

  defp deduplicate(%Result{} = result, seen), do: {result, seen}

  defp deduplicate(%Candidate{} = candidate, seen) do
    key = Candidate.deduplication_key(candidate)

    if MapSet.member?(seen, key) do
      {Result.duplicate(candidate), seen}
    else
      {candidate, MapSet.put(seen, key)}
    end
  end
end
