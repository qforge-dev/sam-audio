defmodule SamSourcePipeline.Stages.Qualification do
  @moduledoc "Applies the real, pure acceptance policy to enriched candidates."

  use GenStage

  alias SamSourcePipeline.Domain.{EnrichedCandidate, Result}
  alias SamSourcePipeline.QualificationPolicy

  def start_link(opts) do
    GenStage.start_link(__MODULE__, opts, name: Keyword.fetch!(opts, :name))
  end

  @impl true
  def init(opts) do
    subscriptions =
      Enum.map(Keyword.fetch!(opts, :producers), fn producer ->
        {producer, Keyword.fetch!(opts, :demand)}
      end)

    {:producer_consumer, :ok, subscribe_to: subscriptions}
  end

  @impl true
  def handle_events(events, _from, state) do
    events = Enum.map(events, &qualify/1)
    {:noreply, events, state}
  end

  defp qualify(%Result{} = result), do: result

  defp qualify(%EnrichedCandidate{} = enriched) do
    case QualificationPolicy.evaluate(enriched) do
      {:accepted, score} ->
        Result.qualified(enriched, :accepted, score, nil)

      {:rejected, reasons, score} ->
        Result.qualified(enriched, :rejected, score, reasons)
    end
  end
end
