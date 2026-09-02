'use strict';

/**
 * Convert the bridge response into Promptfoo output, usage, cost, and timing metadata.
 * @param {Record<string, any>} json Parsed bridge response.
 * @returns {Record<string, any>} Promptfoo ProviderResponse.
 */
module.exports = function parseBridgeResponse(json) {
  const trace = json?.trace ?? {};
  const usage = trace?.usage ?? {};
  return {
    output: String(json?.output ?? ''),
    tokenUsage: {
      prompt: Number(usage.prompt_tokens ?? 0),
      completion: Number(usage.completion_tokens ?? 0),
      total: Number(usage.total_tokens ?? 0),
      numRequests: Array.isArray(trace.rounds) ? trace.rounds.length : 1,
    },
    cost: Number(usage.cost ?? 0),
    metadata: {
      backend: json?.backend ?? '',
      totalDurationSeconds: Number(trace.total_duration_seconds ?? 0),
      toolCallCount: Array.isArray(trace.tool_calls) ? trace.tool_calls.length : 0,
      toolCalls: Array.isArray(trace.tool_calls)
        ? trace.tool_calls.map(({ name, arguments: args }) => ({ name: name ?? '', arguments: args ?? '' }))
        : [],
    },
  };
};
