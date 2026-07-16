defmodule SamSourcePipeline.Domain.Result do
  @moduledoc "A terminal, persistable outcome produced for a request or candidate."

  alias SamSourcePipeline.Domain.{Candidate, EnrichedCandidate, SearchRequest}

  @enforce_keys [:request_id, :status, :stage, :completed_at]
  defstruct [
    :request_id,
    :candidate_id,
    :status,
    :stage,
    :score,
    :value,
    :reason,
    :completed_at
  ]

  @type status :: :accepted | :rejected | :failed
  @type t :: %__MODULE__{
          request_id: String.t(),
          candidate_id: String.t() | nil,
          status: status(),
          stage: atom(),
          score: float() | nil,
          value: EnrichedCandidate.t() | Candidate.t() | SearchRequest.t() | nil,
          reason: term(),
          completed_at: DateTime.t()
        }

  @spec search_failed(SearchRequest.t(), term()) :: t()
  def search_failed(%SearchRequest{} = request, reason) do
    build(request.id, nil, :failed, :search, request, reason)
  end

  @spec candidate_failed(Candidate.t(), atom(), term()) :: t()
  def candidate_failed(%Candidate{} = candidate, stage, reason) do
    build(candidate.request_id, candidate.id, :failed, stage, candidate, reason)
  end

  @spec duplicate(Candidate.t()) :: t()
  def duplicate(%Candidate{} = candidate) do
    build(
      candidate.request_id,
      candidate.id,
      :rejected,
      :deduplication,
      candidate,
      :duplicate_source
    )
  end

  @spec qualified(EnrichedCandidate.t(), :accepted | :rejected, float(), term()) :: t()
  def qualified(%EnrichedCandidate{candidate: candidate} = enriched, status, score, reason) do
    %__MODULE__{
      request_id: candidate.request_id,
      candidate_id: candidate.id,
      status: status,
      stage: :qualification,
      score: score,
      value: enriched,
      reason: reason,
      completed_at: DateTime.utc_now()
    }
  end

  defp build(request_id, candidate_id, status, stage, value, reason) do
    %__MODULE__{
      request_id: request_id,
      candidate_id: candidate_id,
      status: status,
      stage: stage,
      value: value,
      reason: reason,
      completed_at: DateTime.utc_now()
    }
  end
end
