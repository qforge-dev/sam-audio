defmodule SamSourcePipeline.QueryQueueTest do
  use ExUnit.Case, async: true

  alias SamSourcePipeline.Domain.SearchRequest
  alias SamSourcePipeline.QueryQueue

  test "keeps requests queued when no consumer has sent demand" do
    name = String.to_atom("query_queue_#{System.unique_integer([:positive])}")
    start_supervised!({QueryQueue, name: name})

    Enum.each(1..3, fn index ->
      :ok = QueryQueue.enqueue(name, SearchRequest.new("query #{index}"))
    end)

    assert QueryQueue.stats(name) == %{queued: 3, outstanding_demand: 0}
  end
end
