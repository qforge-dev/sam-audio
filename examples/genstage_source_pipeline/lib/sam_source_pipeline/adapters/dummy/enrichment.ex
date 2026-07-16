defmodule SamSourcePipeline.Adapters.Dummy.Enrichment do
  @moduledoc "Deterministic metadata/proxy-analysis stand-in."

  @behaviour SamSourcePipeline.Adapters.Enrichment

  alias SamSourcePipeline.Domain.{Candidate, EnrichedCandidate}

  @profiles [
    %{foreground_speech: 0.78, music: 0.58, sound_effects: 0.46},
    %{foreground_speech: 0.31, music: 0.61, sound_effects: 0.52},
    %{foreground_speech: 0.73, music: 0.12, sound_effects: 0.40},
    %{foreground_speech: 0.65, music: 0.44, sound_effects: 0.37}
  ]

  @impl true
  def enrich(%Candidate{} = candidate) do
    Process.sleep(75)
    rank = candidate.metadata.rank
    profile = Enum.at(@profiles, rem(rank - 1, length(@profiles)))

    {:ok,
     %EnrichedCandidate{
       candidate: candidate,
       duration_seconds: 90 + rank * 31,
       language: if(rem(rank, 3) == 0, do: "es", else: "en"),
       tags: ["cinematic", "dialogue", candidate.provider |> Atom.to_string()],
       audio_profile: profile,
       metadata: %{
         codec: "aac",
         sample_rate: 48_000,
         channels: 2,
         enrichment_adapter: __MODULE__
       }
     }}
  end
end
