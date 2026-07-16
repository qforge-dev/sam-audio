defmodule SamSourcePipeline.QualificationPolicyTest do
  use ExUnit.Case, async: true

  alias SamSourcePipeline.Domain.{Candidate, EnrichedCandidate}
  alias SamSourcePipeline.QualificationPolicy

  test "accepts a source with dialogue, music, and sound-effect evidence" do
    enriched = enriched_candidate(%{foreground_speech: 0.8, music: 0.5, sound_effects: 0.4})

    assert {:accepted, 0.625} = QualificationPolicy.evaluate(enriched)
  end

  test "returns every failed gate for a rejected source" do
    enriched =
      enriched_candidate(
        %{foreground_speech: 0.2, music: 0.1, sound_effects: 0.05},
        duration_seconds: 20
      )

    assert {:rejected, reasons, _score} = QualificationPolicy.evaluate(enriched)

    assert reasons == [
             :source_too_short,
             :insufficient_foreground_speech,
             :insufficient_music,
             :insufficient_sound_effects
           ]
  end

  defp enriched_candidate(profile, opts \\ []) do
    candidate = %Candidate{
      id: "youtube:test",
      request_id: "request-1",
      provider: :youtube,
      external_id: "test",
      url: "https://example.invalid/test",
      title: "Test"
    }

    %EnrichedCandidate{
      candidate: candidate,
      duration_seconds: Keyword.get(opts, :duration_seconds, 120),
      language: "en",
      tags: [],
      audio_profile: profile
    }
  end
end
