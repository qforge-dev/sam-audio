defmodule SamSourcePipeline.Adapters.Dummy.Search do
  @moduledoc "Deterministic stand-in for YouTube, Dailymotion, or catalog search."

  @behaviour SamSourcePipeline.Adapters.Search

  alias SamSourcePipeline.Domain.{Candidate, SearchRequest}

  @impl true
  def search(%SearchRequest{} = request) do
    Process.sleep(40)

    candidates =
      for rank <- 1..request.limit do
        provider = if rem(rank, 2) == 0, do: :dailymotion, else: :youtube
        external_id = external_id(request.query, rank)

        %Candidate{
          id: "#{provider}:#{external_id}",
          request_id: request.id,
          provider: provider,
          external_id: external_id,
          url: "https://example.invalid/#{provider}/#{external_id}",
          title: "#{request.query} — result #{rank}",
          metadata: %{rank: rank, query: request.query, filters: request.filters}
        }
      end

    {:ok, candidates}
  end

  defp external_id(query, rank) do
    query
    |> then(&:crypto.hash(:sha256, "#{&1}:#{rank}"))
    |> Base.url_encode64(padding: false)
    |> binary_part(0, 12)
  end
end
