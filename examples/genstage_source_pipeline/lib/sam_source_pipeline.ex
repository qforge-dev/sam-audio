defmodule SamSourcePipeline do
  @moduledoc "Public command/query API for the supervised GenStage workflow."

  alias SamSourcePipeline.Adapters.Dummy.Repository, as: DummyRepository
  alias SamSourcePipeline.Domain.SearchRequest

  @spec search(String.t(), keyword()) :: {:ok, String.t()} | {:error, String.t()}
  def search(query, opts \\ []) do
    request = SearchRequest.new(query, opts)
    :ok = SamSourcePipeline.QueryQueue.enqueue(request)
    {:ok, request.id}
  rescue
    exception in ArgumentError -> {:error, Exception.message(exception)}
  end

  @spec results(String.t() | nil) :: [SamSourcePipeline.Domain.Result.t()]
  def results(request_id \\ nil) do
    repository().all(repository())
    |> maybe_filter(request_id)
  end

  @spec reset_results() :: :ok
  def reset_results do
    repository().reset(repository())
  end

  @spec queue_stats() :: map()
  def queue_stats do
    SamSourcePipeline.QueryQueue.stats()
  end

  @spec await(String.t(), pos_integer(), timeout()) :: {:ok, list()} | {:error, :timeout}
  def await(request_id, expected_count, timeout \\ 5_000)
      when is_integer(expected_count) and expected_count > 0 do
    deadline = System.monotonic_time(:millisecond) + timeout
    await_results(request_id, expected_count, deadline)
  end

  defp await_results(request_id, expected_count, deadline) do
    results = results(request_id)

    cond do
      length(results) >= expected_count ->
        {:ok, results}

      System.monotonic_time(:millisecond) >= deadline ->
        {:error, :timeout}

      true ->
        Process.sleep(20)
        await_results(request_id, expected_count, deadline)
    end
  end

  defp maybe_filter(results, nil), do: results
  defp maybe_filter(results, request_id), do: Enum.filter(results, &(&1.request_id == request_id))

  defp repository do
    Application.get_env(:sam_source_pipeline, :repository_adapter, DummyRepository)
  end
end
