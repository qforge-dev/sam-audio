defmodule SamSourcePipeline.Application do
  @moduledoc false

  use Application

  @impl true
  def start(_type, _args) do
    children = [SamSourcePipeline.PipelineSupervisor]

    Supervisor.start_link(children,
      strategy: :one_for_one,
      name: SamSourcePipeline.RootSupervisor
    )
  end
end
