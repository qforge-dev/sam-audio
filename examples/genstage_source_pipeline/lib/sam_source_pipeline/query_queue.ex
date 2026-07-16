defmodule SamSourcePipeline.QueryQueue do
  @moduledoc "In-memory FIFO producer that releases search requests only on demand."

  use GenStage

  alias SamSourcePipeline.Domain.SearchRequest

  def start_link(opts) do
    GenStage.start_link(__MODULE__, :ok, name: Keyword.fetch!(opts, :name))
  end

  @spec enqueue(GenServer.server(), SearchRequest.t()) :: :ok
  def enqueue(server \\ __MODULE__, %SearchRequest{} = request) do
    GenStage.call(server, {:enqueue, request})
  end

  @spec stats(GenServer.server()) :: %{
          queued: non_neg_integer(),
          outstanding_demand: non_neg_integer()
        }
  def stats(server \\ __MODULE__) do
    GenStage.call(server, :stats)
  end

  @impl true
  def init(:ok) do
    {:producer, %{queue: :queue.new(), demand: 0}}
  end

  @impl true
  def handle_demand(incoming, state) when incoming > 0 do
    {events, state} = dispatch(%{state | demand: state.demand + incoming})
    {:noreply, events, state}
  end

  @impl true
  def handle_call({:enqueue, request}, _from, state) do
    state = %{state | queue: :queue.in(request, state.queue)}
    {events, state} = dispatch(state)
    {:reply, :ok, events, state}
  end

  @impl true
  def handle_call(:stats, _from, state) do
    stats = %{queued: :queue.len(state.queue), outstanding_demand: state.demand}
    {:reply, stats, [], state}
  end

  defp dispatch(state), do: dispatch(state, [])

  defp dispatch(%{demand: 0} = state, events), do: {Enum.reverse(events), state}

  defp dispatch(state, events) do
    case :queue.out(state.queue) do
      {{:value, request}, queue} ->
        state = %{state | queue: queue, demand: state.demand - 1}
        dispatch(state, [request | events])

      {:empty, _queue} ->
        {Enum.reverse(events), state}
    end
  end
end
