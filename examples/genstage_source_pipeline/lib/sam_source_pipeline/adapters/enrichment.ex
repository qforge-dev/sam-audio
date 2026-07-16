defmodule SamSourcePipeline.Adapters.Enrichment do
  @moduledoc "Port for metadata lookup and lightweight source analysis."

  alias SamSourcePipeline.Domain.{Candidate, EnrichedCandidate}

  @callback enrich(Candidate.t()) :: {:ok, EnrichedCandidate.t()} | {:error, term()}
end
