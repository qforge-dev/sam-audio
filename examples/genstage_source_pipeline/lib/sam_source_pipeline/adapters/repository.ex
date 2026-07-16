defmodule SamSourcePipeline.Adapters.Repository do
  @moduledoc "Port for terminal result persistence."

  alias SamSourcePipeline.Domain.Result

  @callback child_spec(keyword()) :: Supervisor.child_spec()
  @callback save(GenServer.server(), Result.t()) :: :ok | {:error, term()}
  @callback all(GenServer.server()) :: [Result.t()]
  @callback reset(GenServer.server()) :: :ok
end
