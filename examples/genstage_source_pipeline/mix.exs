defmodule SamSourcePipeline.MixProject do
  use Mix.Project

  def project do
    [
      app: :sam_source_pipeline,
      version: "0.1.0",
      elixir: "~> 1.15",
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  def application do
    [
      extra_applications: [:logger],
      mod: {SamSourcePipeline.Application, []}
    ]
  end

  defp deps do
    [
      {:gen_stage, "~> 1.3"}
    ]
  end
end
