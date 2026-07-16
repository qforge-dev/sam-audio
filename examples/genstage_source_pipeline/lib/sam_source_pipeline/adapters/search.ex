defmodule SamSourcePipeline.Adapters.Search do
  @moduledoc "Port implemented by a real search provider or the demo adapter."

  alias SamSourcePipeline.Domain.{Candidate, SearchRequest}

  @callback search(SearchRequest.t()) :: {:ok, [Candidate.t()]} | {:error, term()}
end
