defmodule SamSourcePipeline.Domain.Candidate do
  @moduledoc "A source returned by a search provider before enrichment."

  @enforce_keys [:id, :request_id, :provider, :external_id, :url, :title]
  defstruct [:id, :request_id, :provider, :external_id, :url, :title, metadata: %{}]

  @type t :: %__MODULE__{
          id: String.t(),
          request_id: String.t(),
          provider: atom(),
          external_id: String.t(),
          url: String.t(),
          title: String.t(),
          metadata: map()
        }

  @spec deduplication_key(t()) :: {atom(), String.t()}
  def deduplication_key(%__MODULE__{} = candidate) do
    {candidate.provider, candidate.external_id}
  end
end
