defmodule SamSourcePipeline.AdapterCall do
  @moduledoc false

  @spec run((-> term())) :: {:ok, term()} | {:error, term()}
  def run(fun) when is_function(fun, 0) do
    case fun.() do
      {:ok, value} -> {:ok, value}
      {:error, reason} -> {:error, reason}
      other -> {:error, {:invalid_adapter_response, other}}
    end
  rescue
    exception -> {:error, {:exception, Exception.message(exception)}}
  catch
    kind, reason -> {:error, {kind, reason}}
  end
end
