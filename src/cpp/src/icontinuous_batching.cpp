// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "icontinuous_batching.hpp"

namespace ov::genai {

GenerationConfig ContinuousBatchingPipeline::IContinuousBatchingPipeline::get_config() const {
    return m_generation_config;
}

PipelineMetrics ContinuousBatchingPipeline::IContinuousBatchingPipeline::get_metrics() const {
    return m_pipeline_metrics;
}

Tokenizer ContinuousBatchingPipeline::IContinuousBatchingPipeline::get_tokenizer() {
    return m_tokenizer;
}

void ContinuousBatchingPipeline::IContinuousBatchingPipeline::start_chat(const std::string& system_message) {
    if (!system_message.empty()) {
        m_history.push_back({{"role", "system"}, {"content", system_message}});
    }
    m_is_chat_conversation = true;
};

void ContinuousBatchingPipeline::IContinuousBatchingPipeline::finish_chat() {
    m_is_chat_conversation = false;
    m_history.clear();
};

std::vector<GenerationResult>
ContinuousBatchingPipeline::IContinuousBatchingPipeline::generate(
    const std::vector<std::string>& prompts,
    std::vector<ov::genai::GenerationConfig> sampling_params,
    const StreamerVariant& streamer) {
    std::vector<ov::Tensor> input_ids;
    auto start_time =  std::chrono::steady_clock::now();

    std::vector<MicroSeconds> tokenization_durations;
    static ManualTimer timer("tokenize");
    if (m_is_chat_conversation) {
        OPENVINO_ASSERT(1 == prompts.size(), "Can't chat with multiple prompts");
        m_history.push_back({{"role", "user"}, {"content", prompts.at(0)}});
        constexpr bool add_generation_prompt = true;
        std::string history = m_tokenizer.apply_chat_template(m_history, add_generation_prompt);
        timer.start();
        const auto encode_start = std::chrono::steady_clock::now();
        // ov::genai::add_special_tokens(false) is aligned with stateful pipeline
        input_ids.push_back(m_tokenizer.encode(history, ov::genai::add_special_tokens(false)).input_ids);
        tokenization_durations.emplace_back(PerfMetrics::get_microsec(std::chrono::steady_clock::now() - encode_start));
        timer.end();
    } else {
        input_ids.reserve(prompts.size());
        timer.start();
        for (const std::string& prompt : prompts) {
            const auto encode_start = std::chrono::steady_clock::now();
            input_ids.push_back(m_tokenizer.encode(prompt).input_ids);
            tokenization_durations.emplace_back(PerfMetrics::get_microsec(std::chrono::steady_clock::now() - encode_start));
        }
        timer.end();
    }

    std::vector<EncodedGenerationResult> encoded = generate(input_ids, sampling_params, streamer);

    std::vector<GenerationResult> decoded;
    decoded.reserve(encoded.size());
    for (size_t i = 0; i < encoded.size(); ++i) {
        EncodedGenerationResult res = encoded[i];
        auto& perf_metrics = res.perf_metrics;
        auto& raw_counters = perf_metrics.raw_metrics;
        raw_counters.tokenization_durations.emplace_back(tokenization_durations[i]);

        std::vector<std::string> generated;
        generated.reserve(res.m_generation_ids.size());
        for (size_t idx = 0; idx < res.m_generation_ids.size(); ++idx) {
            const auto decode_start = std::chrono::steady_clock::now();
            generated.push_back(m_tokenizer.decode(res.m_generation_ids.at(idx)));
            raw_counters.detokenization_durations.emplace_back(std::chrono::steady_clock::now() - decode_start);
            if (m_is_chat_conversation && 0 == idx) {
                m_history.push_back({{"role", "assistant"}, {"content", generated.back()}});
            }
        }

        // The same perf metrics for each sequence, only tokenization/detokenization will differ.
        perf_metrics.raw_metrics.generate_durations.clear();
        perf_metrics.raw_metrics.generate_durations.emplace_back(PerfMetrics::get_microsec(std::chrono::steady_clock::now() - start_time));
        // Reevaluate taking into accound tokenization/detokenization times.
        perf_metrics.m_evaluated = false;
        perf_metrics.evaluate_statistics(start_time);

        decoded.push_back(GenerationResult{
            res.m_request_id,
            std::move(generated),
            std::move(res.m_scores),
            res.m_status,
            perf_metrics,
        });
    }

    return decoded;
}
}
