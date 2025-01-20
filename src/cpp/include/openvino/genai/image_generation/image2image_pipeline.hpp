// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/image_generation/inpainting_pipeline.hpp"

namespace ov {
namespace genai {

//
// Image to image pipeline
//

class OPENVINO_GENAI_EXPORTS Image2ImagePipeline {
public:
    explicit Image2ImagePipeline(const std::filesystem::path& models_path);

    Image2ImagePipeline(const std::filesystem::path& models_path, const std::string& device, const ov::AnyMap& properties = {});

    template <typename... Properties,
              typename std::enable_if<ov::util::StringAny<Properties...>::value, bool>::type = true>
    Image2ImagePipeline(const std::filesystem::path& models_path,
                        const std::string& device,
                        Properties&&... properties)
        : Image2ImagePipeline(models_path, device, ov::AnyMap{std::forward<Properties>(properties)...}) { }

    Image2ImagePipeline(const InpaintingPipeline& pipe);

    // creates either LCM or SD pipeline from building blocks
    static Image2ImagePipeline stable_diffusion(
        const std::shared_ptr<Scheduler>& scheduler_type,
        const CLIPTextModel& clip_text_model,
        const UNet2DConditionModel& unet,
        const AutoencoderKL& vae);

    // creates either LCM or SD pipeline from building blocks
    static Image2ImagePipeline latent_consistency_model(
        const std::shared_ptr<Scheduler>& scheduler_type,
        const CLIPTextModel& clip_text_model,
        const UNet2DConditionModel& unet,
        const AutoencoderKL& vae);

    // creates SDXL pipeline from building blocks
    static Image2ImagePipeline stable_diffusion_xl(
        const std::shared_ptr<Scheduler>& scheduler_type,
        const CLIPTextModel& clip_text_model,
        const CLIPTextModelWithProjection& clip_text_model_with_projection,
        const UNet2DConditionModel& unet,
        const AutoencoderKL& vae);

    ImageGenerationConfig get_generation_config() const;
    void set_generation_config(const ImageGenerationConfig& generation_config);

    // ability to override scheduler
    void set_scheduler(std::shared_ptr<Scheduler> scheduler);

    // with static shapes performance is better
    void reshape(const int num_images_per_prompt, const int height, const int width, const float guidance_scale);

    void compile(const std::string& device, const ov::AnyMap& properties = {});

    template <typename... Properties>
    ov::util::EnableIfAllStringAny<void, Properties...> compile(
            const std::string& device,
            Properties&&... properties) {
        return compile(device, ov::AnyMap{std::forward<Properties>(properties)...});
    }

    /**
     * Peforms initial image editing conditioned on a text prompt.
     * @param positive_prompt Prompt to generate image(s) from
     * @param initial_image RGB/BGR image of [1, height, width, 3] shape used to initialize latent image
     * @param properties Image generation parameters specified as properties. Values in 'properties' override default value for generation parameters.
     * @returns A tensor which has dimensions [num_images_per_prompt, height, width, 3]
     * @note Output image size is the same as initial image size, but rounded down to be divisible by VAE scale factor (usually, 8)
     */
    ov::Tensor generate(const std::string& positive_prompt, ov::Tensor initial_image, const ov::AnyMap& properties = {});

    template <typename... Properties>
    ov::util::EnableIfAllStringAny<ov::Tensor, Properties...> generate(
            const std::string& positive_prompt,
            ov::Tensor initial_image,
            Properties&&... properties) {
        return generate(positive_prompt, initial_image, ov::AnyMap{std::forward<Properties>(properties)...});
    }

    ov::Tensor decode(const ov::Tensor latent);

private:
    std::shared_ptr<DiffusionPipeline> m_impl;

    explicit Image2ImagePipeline(const std::shared_ptr<DiffusionPipeline>& impl);

    // to create other pipelines from image to image
    friend class Text2ImagePipeline;
    friend class InpaintingPipeline;
};

} // namespace genai
} // namespace ov
