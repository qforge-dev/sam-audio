defmodule SamSourcePipelineTest do
  use ExUnit.Case, async: false

  setup do
    :ok = SamSourcePipeline.reset_results()
  end

  test "runs search, enrichment, qualification, and persistence end to end" do
    query = "cinematic harbor #{System.unique_integer([:positive])}"
    assert {:ok, request_id} = SamSourcePipeline.search(query, limit: 4)
    assert {:ok, results} = SamSourcePipeline.await(request_id, 4)

    assert Enum.all?(results, &(&1.request_id == request_id))
    assert Enum.count(results, &(&1.status == :accepted)) == 2
    assert Enum.count(results, &(&1.status == :rejected)) == 2
    assert Enum.all?(results, &(&1.stage == :qualification))
    assert Enum.all?(results, &is_float(&1.score))
  end

  test "deduplicates a provider source before running enrichment again" do
    query = "duplicate city #{System.unique_integer([:positive])}"

    assert {:ok, first_request} = SamSourcePipeline.search(query, limit: 2)
    assert {:ok, _results} = SamSourcePipeline.await(first_request, 2)

    assert {:ok, second_request} = SamSourcePipeline.search(query, limit: 2)
    assert {:ok, duplicate_results} = SamSourcePipeline.await(second_request, 2)

    assert Enum.all?(duplicate_results, &(&1.status == :rejected))
    assert Enum.all?(duplicate_results, &(&1.stage == :deduplication))
    assert Enum.all?(duplicate_results, &(&1.reason == :duplicate_source))
  end

  test "validates commands before adding them to the queue" do
    assert {:error, "search query cannot be empty"} = SamSourcePipeline.search("  ")
    assert {:error, "limit must be positive"} = SamSourcePipeline.search("valid", limit: 0)
  end
end
