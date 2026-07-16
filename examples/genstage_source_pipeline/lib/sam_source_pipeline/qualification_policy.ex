defmodule SamSourcePipeline.QualificationPolicy do
  @moduledoc "Pure domain policy deciding whether an enriched source is usable."

  alias SamSourcePipeline.Domain.EnrichedCandidate

  @minimum_duration_seconds 60
  @minimum_foreground_speech 0.45
  @minimum_music 0.20
  @minimum_sound_effects 0.20

  @type decision ::
          {:accepted, float()}
          | {:rejected, [atom()], float()}

  @spec evaluate(EnrichedCandidate.t()) :: decision()
  def evaluate(%EnrichedCandidate{} = enriched) do
    profile = enriched.audio_profile

    score =
      0.50 * profile.foreground_speech +
        0.25 * profile.music +
        0.25 * profile.sound_effects

    reasons =
      []
      |> maybe_reject(
        enriched.duration_seconds < @minimum_duration_seconds,
        :source_too_short
      )
      |> maybe_reject(
        profile.foreground_speech < @minimum_foreground_speech,
        :insufficient_foreground_speech
      )
      |> maybe_reject(profile.music < @minimum_music, :insufficient_music)
      |> maybe_reject(
        profile.sound_effects < @minimum_sound_effects,
        :insufficient_sound_effects
      )
      |> Enum.reverse()

    score = Float.round(score, 4)

    case reasons do
      [] -> {:accepted, score}
      reasons -> {:rejected, reasons, score}
    end
  end

  defp maybe_reject(reasons, true, reason), do: [reason | reasons]
  defp maybe_reject(reasons, false, _reason), do: reasons
end
