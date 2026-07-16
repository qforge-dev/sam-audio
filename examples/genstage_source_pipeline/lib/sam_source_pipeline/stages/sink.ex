defmodule SamSourcePipeline.Stages.Sink do
  @moduledoc "Terminal consumer that persists every accepted, rejected, or failed result."

  use GenStage

  require Logger

  alias SamSourcePipeline.Domain.Result

  def start_link(opts) do
    GenStage.start_link(__MODULE__, opts, name: Keyword.fetch!(opts, :name))
  end

  @impl true
  def init(opts) do
    state = %{
      repository: Keyword.fetch!(opts, :repository),
      repository_server: Keyword.fetch!(opts, :repository_server)
    }

    producer = Keyword.fetch!(opts, :producer)
    demand = Keyword.fetch!(opts, :demand)

    {:consumer, state, subscribe_to: [{producer, demand}]}
  end

  @impl true
  def handle_events(results, _from, state) do
    Enum.each(results, &persist(&1, state))
    {:noreply, [], state}
  end

  defp persist(%Result{} = result, state) do
    case state.repository.save(state.repository_server, result) do
      :ok ->
        Logger.info("stored #{result.status} result",
          stage: :sink,
          request_id: result.request_id,
          candidate_id: result.candidate_id
        )

      {:error, reason} ->
        Logger.error("result persistence failed: #{inspect(reason)}",
          stage: :sink,
          request_id: result.request_id,
          candidate_id: result.candidate_id
        )
    end
  end
end
