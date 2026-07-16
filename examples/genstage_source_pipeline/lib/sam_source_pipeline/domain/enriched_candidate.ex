defmodule SamSourcePipeline.Domain.EnrichedCandidate do
  @moduledoc "Provider metadata and proxy-analysis evidence attached to a candidate."

  alias SamSourcePipeline.Domain.Candidate

  @enforce_keys [:candidate, :duration_seconds, :language, :tags, :audio_profile]
  defstruct [:candidate, :duration_seconds, :language, :tags, :audio_profile, metadata: %{}]

  @type t :: %__MODULE__{
          candidate: Candidate.t(),
          duration_seconds: non_neg_integer(),
          language: String.t() | nil,
          tags: [String.t()],
          audio_profile: %{
            required(:foreground_speech) => float(),
            required(:music) => float(),
            required(:sound_effects) => float()
          },
          metadata: map()
        }
end
