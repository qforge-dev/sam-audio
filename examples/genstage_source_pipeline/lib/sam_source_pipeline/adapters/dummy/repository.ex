defmodule SamSourcePipeline.Adapters.Dummy.Repository do
  @moduledoc "In-memory repository used by the runnable workflow and tests."

  use Agent

  @behaviour SamSourcePipeline.Adapters.Repository

  alias SamSourcePipeline.Domain.Result

  def start_link(opts) do
    Agent.start_link(fn -> [] end, name: Keyword.fetch!(opts, :name))
  end

  @impl true
  def child_spec(opts) do
    name = Keyword.fetch!(opts, :name)

    %{
      id: name,
      start: {__MODULE__, :start_link, [opts]},
      type: :worker,
      restart: :permanent
    }
  end

  @impl true
  def save(server, %Result{} = result) do
    Agent.update(server, &[result | &1])
  end

  @impl true
  def all(server) do
    Agent.get(server, &Enum.reverse/1)
  end

  @impl true
  def reset(server) do
    Agent.update(server, fn _results -> [] end)
  end
end
