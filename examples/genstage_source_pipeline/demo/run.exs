alias SamSourcePipeline.Domain.Result

:ok = SamSourcePipeline.reset_results()

IO.puts("Submitting a real asynchronous search command...\n")

{:ok, first_request} =
  SamSourcePipeline.search("cinematic city dialogue", limit: 4, filters: %{language: "en"})

{:ok, first_results} = SamSourcePipeline.await(first_request, 4)

IO.puts("\nSubmitting the same search again to exercise global deduplication...\n")

{:ok, second_request} = SamSourcePipeline.search("cinematic city dialogue", limit: 4)
{:ok, second_results} = SamSourcePipeline.await(second_request, 4)

print_result = fn %Result{} = result ->
  IO.puts(
    "#{result.status |> Atom.to_string() |> String.upcase()} " <>
      "stage=#{result.stage} candidate=#{result.candidate_id || "none"} " <>
      "score=#{result.score || "n/a"} reason=#{inspect(result.reason)}"
  )
end

IO.puts("\nFirst request results:")
Enum.each(first_results, print_result)

IO.puts("\nDuplicate request results:")
Enum.each(second_results, print_result)

IO.puts("\nQueue state: #{inspect(SamSourcePipeline.queue_stats())}")
