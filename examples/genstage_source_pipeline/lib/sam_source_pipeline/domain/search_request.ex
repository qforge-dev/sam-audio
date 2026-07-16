defmodule SamSourcePipeline.Domain.SearchRequest do
  @moduledoc "The command accepted by the source-discovery pipeline."

  @enforce_keys [:id, :query, :limit, :requested_at]
  defstruct [:id, :query, :limit, :requested_at, filters: %{}]

  @type t :: %__MODULE__{
          id: String.t(),
          query: String.t(),
          limit: pos_integer(),
          requested_at: DateTime.t(),
          filters: map()
        }

  @spec new(String.t(), keyword()) :: t()
  def new(query, opts \\ []) when is_binary(query) do
    query = String.trim(query)
    limit = Keyword.get(opts, :limit, 5)

    if query == "", do: raise(ArgumentError, "search query cannot be empty")
    if not (is_integer(limit) and limit > 0), do: raise(ArgumentError, "limit must be positive")

    %__MODULE__{
      id: request_id(),
      query: query,
      limit: limit,
      requested_at: DateTime.utc_now(),
      filters: Map.new(Keyword.get(opts, :filters, %{}))
    }
  end

  defp request_id do
    suffix = System.unique_integer([:positive, :monotonic])
    "search-#{System.system_time(:millisecond)}-#{suffix}"
  end
end
