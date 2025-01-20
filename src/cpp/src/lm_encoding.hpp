#pragma once

#include <optional>
#include "openvino/genai/llm_pipeline.hpp"
#include "visual_language/embedding_model.hpp"
#include "sampler.hpp"

namespace ov {
namespace genai {

std::pair<EncodedResults, std::optional<int64_t>> get_lm_encoded_results(ov::InferRequest& m_llm, const ov::Tensor& input_ids, const ov::Tensor& attention_mask,
                                                                         const std::shared_ptr<StreamerBase>& streamer_ptr, Sampler& sampler, std::vector<SequenceGroup::Ptr> sequence_groups,
                                                                         std::optional<ov::Tensor> position_ids, std::optional<EmbeddingsModel> m_embedding, std::optional<int64_t> rope_delta = std::nullopt);

}
}
