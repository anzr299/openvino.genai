// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>
#include <filesystem>

#include "openvino/genai/tokenizer.hpp"
#include "openvino/genai/visual_language/pipeline.hpp"
#include "openvino/runtime/tensor.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "visual_language/vlm_config.hpp"
#include "visual_language/embedding_model.hpp"

namespace ov::genai {
struct VLMPerfMetrics;

class InputsEmbedder {
public:
    InputsEmbedder(const VLMConfig& vlm_config,
                   const std::filesystem::path& model_dir,
                   const std::string& device,
                   const ov::AnyMap device_config);

    InputsEmbedder(const VLMConfig& vlm_config,
                   const ModelsMap& models_map,
                   const Tokenizer& tokenizer,
                   const std::filesystem::path& config_dir_path,
                   const std::string& device,
                   const ov::AnyMap device_config);

    // compute input embedding for prompt and multiple images
    ov::Tensor get_inputs_embeds(const std::string& prompt, const std::vector<ov::Tensor>& images, ov::genai::VLMPerfMetrics& metrics);

    // compute position ids for language model input
    std::pair<ov::Tensor, std::optional<int64_t>> get_position_ids(const size_t inputs_embeds_size, const size_t history_size);

    // returns embedding model which converts token_id(s) to embedding vectors
    EmbeddingsModel get_embedding_model() const;

    // returns tokenizer
    Tokenizer get_tokenizer() const;

    // returns tokenized chat history
    std::vector<int64_t> get_tokenized_history() const;

    // add new results to tokenized history
    void update_tokenized_history(const std::vector<int64_t>& encoded_result, std::optional<int64_t> last_disappeared_token, bool is_beam_search, size_t last_answer_len);

    // returns amount of elements, which need to remove from the end of the KV cache
    size_t get_num_tokens_to_remove_from_hist() const;

    // starts chat and adds optional system_message to chat history
    void start_chat(const std::string& system_message);

    // adds currently generated text to chat history
    void update_chat_history(const std::string& decoded_results);

    // finishes chat and clears a chat history 
    void finish_chat();
private:
    class IInputsEmbedder;
    std::shared_ptr<IInputsEmbedder> m_impl;

    friend class InputsEmbedderMiniCPM;
    friend class InputsEmbedderLLaVA;
    friend class InputsEmbedderLLaVANext;
    friend class InputsEmbedderInternVLChat;
    friend class InputsEmbedderQwen2VL;
};

} // namespace ov::genai
