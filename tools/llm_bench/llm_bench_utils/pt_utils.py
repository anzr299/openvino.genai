# -*- coding: utf-8 -*-
# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import torch
from llm_bench_utils.config_class import PT_MODEL_CLASSES_MAPPING, TOKENIZE_CLASSES_MAPPING, DEFAULT_MODEL_CLASSES
import os
import time
import logging as log
import llm_bench_utils.hook_common as hook_common
import json
from nncf.torch.dynamic_graph.patch_pytorch import disable_patching
import nncf
import pickle
from nncf.quantization.range_estimator import RangeEstimatorParametersSet
from diffusers.configuration_utils import ConfigMixin, register_to_config
import numpy as np
import torch
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from transformers.models.clip import CLIPTextModelWithProjection
from diffusers import StableDiffusion3Pipeline, ModelMixin

# This function takes in the models of a SD3 pipeline in the torch fx representation and returns an SD3 pipeline with wrapped models.
def init_pipeline(models_dict, configs_dict, model_id="stabilityai/stable-diffusion-3-medium-diffusers"):
    wrapped_models = {}

    def wrap_model(pipe_model, base_class, config):
        base_class = (base_class,) if not isinstance(base_class, tuple) else base_class

        class WrappedModel(*base_class):
            def __init__(self, model, config):
                self.cls_name = base_class[0].__name__
                if isinstance(config, dict):
                    super().__init__(**config)
                else:
                    super().__init__(config)
                if self.cls_name == "AutoencoderKL":
                    self.encoder = model.encoder
                    self.decoder = model.decoder
                else:
                    self.text_model = model

            def forward(self, *args, **kwargs):
                if self.cls_name == "AutoencoderKL":
                    return self.model(*args, **kwargs)
                else:
                    return self.text_model(*args, **kwargs)

        class WrappedTransformer(*base_class):
            @register_to_config
            def __init__(
                self,
                model,
                sample_size,
                patch_size,
                in_channels,
                num_layers,
                attention_head_dim,
                num_attention_heads,
                joint_attention_dim,
                caption_projection_dim,
                pooled_projection_dim,
                out_channels,
                pos_embed_max_size,
                dual_attention_layers,
                qk_norm,
            ):
                super().__init__()
                self.model = model

            def forward(self, *args, **kwargs):
                del kwargs["joint_attention_kwargs"]
                del kwargs["return_dict"]
                return self.model(*args, **kwargs)

        if len(base_class) > 1:
            return WrappedTransformer(pipe_model, **config)
        return WrappedModel(pipe_model, config)

    wrapped_models["transformer"] = wrap_model(
        models_dict["transformer"],
        (
            ModelMixin,
            ConfigMixin,
        ),
        configs_dict["transformer"],
    )
    wrapped_models["vae"] = wrap_model(models_dict["vae"], AutoencoderKL, configs_dict["vae"])
    wrapped_models["text_encoder"] = wrap_model(models_dict["text_encoder"], CLIPTextModelWithProjection, configs_dict["text_encoder"])
    wrapped_models["text_encoder_2"] = wrap_model(models_dict["text_encoder_2"], CLIPTextModelWithProjection, configs_dict["text_encoder_2"])

    pipe = StableDiffusion3Pipeline.from_pretrained(model_id, text_encoder_3=None, tokenizer_3=None, **wrapped_models)

    return pipe

def get_model_inputs():
    text_encoder_input = torch.ones((1, 77), dtype=torch.long)
    text_encoder_kwargs = {}
    text_encoder_kwargs["output_hidden_states"] = True

    vae_encoder_input = torch.ones((1, 3, 128, 128))
    vae_decoder_input = torch.ones((1, 16, 128, 128))

    unet_kwargs = {}
    unet_kwargs["hidden_states"] = torch.ones((2, 16, 128, 128))
    unet_kwargs["timestep"] = torch.from_numpy(np.array([1, 2], dtype=np.float32))
    unet_kwargs["encoder_hidden_states"] = torch.ones((2, 154, 4096))
    unet_kwargs["pooled_projections"] = torch.ones((2, 2048))
    return text_encoder_input, text_encoder_kwargs, vae_encoder_input, vae_decoder_input, unet_kwargs

def export_models(pipe):
    configs_dict = {}
    configs_dict["text_encoder"] = pipe.text_encoder.config 
    configs_dict["text_encoder_2"] = pipe.text_encoder_2.config
    configs_dict["transformer"] = pipe.transformer.config
    configs_dict["vae"] = pipe.vae.config
    text_encoder_input, text_encoder_kwargs, vae_encoder_input, vae_decoder_input, unet_kwargs = get_model_inputs()

    with torch.no_grad():
        with disable_patching():
            text_encoder = torch.export.export_for_training(
                pipe.text_encoder.eval(),
                args=(text_encoder_input,),
                kwargs=(text_encoder_kwargs),
            ).module()
            text_encoder_2 = torch.export.export_for_training(
                pipe.text_encoder_2.eval(),
                args=(text_encoder_input,),
                kwargs=(text_encoder_kwargs),
            ).module()
            pipe.vae.decoder = torch.export.export_for_training(pipe.vae.decoder.eval(), args=(vae_decoder_input,)).module()
            pipe.vae.encoder = torch.export.export_for_training(pipe.vae.encoder.eval(), args=(vae_encoder_input,)).module()
            vae = pipe.vae
            transformer = torch.export.export_for_training(pipe.transformer.eval(), args=(), kwargs=(unet_kwargs)).module()
    models_dict = {}
    models_dict["transformer"] = transformer
    models_dict["vae"] = vae
    models_dict["text_encoder"] = text_encoder
    models_dict["text_encoder_2"] = text_encoder_2
    return models_dict, configs_dict

def set_bf16(model, device, **kwargs):
    try:
        if len(kwargs['config']) > 0 and kwargs['config'].get('PREC_BF16') and kwargs['config']['PREC_BF16'] is True:
            model = model.to(device.lower(), dtype=torch.bfloat16)
            log.info('Set inference precision to bf16')
    except Exception:
        log.error('Catch exception for setting inference precision to bf16.')
        raise RuntimeError('Set prec_bf16 fail.')
    return model


def torch_compile_child_module(model, child_modules, backend='openvino', dynamic=None, options=None):
    if len(child_modules) == 1:
        setattr(model, child_modules[0], torch.compile(getattr(model, child_modules[0]), backend=backend, dynamic=dynamic, fullgraph=True, options=options))
        return model
    setattr(model, child_modules[0], torch_compile_child_module(getattr(model, child_modules[0]), child_modules[1:], backend, dynamic, options))
    return model


def run_torch_compile(model, backend='openvino', dynamic=None, options=None, child_modules=None):
    if backend == 'pytorch':
        log.info(f'Running torch.compile() with {backend} backend')
        start = time.perf_counter()
        compiled_model = torch.compile(model)
        end = time.perf_counter()
        compile_time = end - start
        log.info(f'Compiling model via torch.compile() took: {compile_time}')
    else:
        log.info(f'Running torch.compile() with {backend} backend')
        start = time.perf_counter()
        if child_modules and len(child_modules) > 0:
            compiled_model = torch_compile_child_module(model, child_modules, backend, dynamic, options)
        else:
            options = {"device" : "CPU", "config" : {"PERFORMANCE_HINT" : "LATENCY"}}
            compiled_model = torch.compile(model, backend=backend, dynamic=dynamic, options=options)
        end = time.perf_counter()
        compile_time = end - start
        log.info(f'Compiling model via torch.compile() took: {compile_time}')
    return compiled_model


def create_text_gen_model(model_path, device, **kwargs):
    model_path = Path(model_path)
    from_pretrain_time = 0
    if model_path.exists():
        if model_path.is_dir() and len(os.listdir(model_path)) != 0:
            log.info(f'Load text model from model path:{model_path}')
            default_model_type = DEFAULT_MODEL_CLASSES[kwargs['use_case']]
            model_type = kwargs.get('model_type', default_model_type)
            model_class = PT_MODEL_CLASSES_MAPPING.get(model_type, PT_MODEL_CLASSES_MAPPING[default_model_type])
            token_class = TOKENIZE_CLASSES_MAPPING.get(model_type, TOKENIZE_CLASSES_MAPPING[default_model_type])
            start = time.perf_counter()
            trust_remote_code = False
            try:
                model = model_class.from_pretrained(model_path, trust_remote_code=trust_remote_code)
            except Exception:
                start = time.perf_counter()
                trust_remote_code = True
                model = model_class.from_pretrained(model_path, trust_remote_code=trust_remote_code)
            tokenizer = token_class.from_pretrained(model_path, trust_remote_code=trust_remote_code)
            end = time.perf_counter()
            from_pretrain_time = end - start
        else:
            raise RuntimeError(f'==Failure ==: model path:{model_path} is not directory or directory is empty')
    else:
        raise RuntimeError(f'==Failure ==: model path:{model_path} is not exist')

    log.info(f'model path:{model_path}, from pretrained time: {from_pretrain_time:.2f}s')

    if device is not None:
        gptjfclm = 'transformers.models.gptj.modeling_gptj.GPTJForCausalLM'
        lfclm = 'transformers.models.llama.modeling_llama.LlamaForCausalLM'
        bfclm = 'transformers.models.bloom.modeling_bloom.BloomForCausalLM'
        gpt2lmhm = 'transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel'
        gptneoxclm = 'transformers.models.gpt_neox.modeling_gpt_neox.GPTNeoXForCausalLM'
        chatglmfcg = 'transformers_modules.pytorch_original.modeling_chatglm.ChatGLMForConditionalGeneration'
        real_base_model_name = str(type(model)).lower()
        log.info(f'Real base model={real_base_model_name}')
        # bfclm will trigger generate crash.

        # If the device is set to GPU there's a need to substitute it with 'cuda' so it will be accepted by PyTorch
        if device.upper() == 'GPU':
            device = torch.device('cuda') if torch.cuda.is_available() else log.info('CUDA device is unavailable')
        else:
            device = torch.device(device.lower())
        log.info(f'Torch device was set to: {device}')

        if any(x in real_base_model_name for x in [gptjfclm, lfclm, bfclm, gpt2lmhm, gptneoxclm, chatglmfcg]):
            model = set_bf16(model, device, **kwargs)
        else:
            if len(kwargs['config']) > 0 and kwargs['config'].get('PREC_BF16') and kwargs['config']['PREC_BF16'] is True:
                log.info('Param [bf16/prec_bf16] will not work.')
            model.to(device)
    else:
        raise RuntimeError('==Failure ==: no device to load')

    bench_hook = hook_common.get_bench_hook(kwargs['num_beams'], model)

    if kwargs['torch_compile_backend']:
        backend = kwargs['torch_compile_backend']
        dynamic = None
        options = None
        child_modules = None
        if kwargs['torch_compile_dynamic']:
            dynamic = kwargs['torch_compile_dynamic']
        if kwargs['torch_compile_options']:
            options = json.loads(kwargs['torch_compile_options'])
        if kwargs['torch_compile_input_module']:
            child_modules = kwargs['torch_compile_input_module'].split(".")
        compiled_model = run_torch_compile(model, backend, dynamic, options, child_modules)
        model = compiled_model
    return model, tokenizer, from_pretrain_time, bench_hook, False


def create_image_gen_model(model_path, device, **kwargs):
    model_path = Path(model_path)
    from_pretrain_time = 0
    if model_path.exists():
        if model_path.is_dir() and len(os.listdir(model_path)) != 0:
            log.info(f'Load image model from model path:{model_path}')
            model_type = DEFAULT_MODEL_CLASSES[kwargs['use_case']]
            model_class = PT_MODEL_CLASSES_MAPPING[model_type]
            start = time.perf_counter()
            pipe = model_class.from_pretrained(model_path, text_encoder_3=None, tokenizer_3=None)
            pipe = set_bf16(pipe, device, **kwargs)
            end = time.perf_counter()
            from_pretrain_time = end - start
        else:
            raise RuntimeError(f'==Failure ==: model path:{model_path} is not directory or directory is empty')
    else:
        raise RuntimeError(f'==Failure ==: model path:{model_path} is not exist')

    log.info(f'Model path:{model_path}, from pretrained time: {from_pretrain_time:.2f}s')

    if device:
        # If the device is set to GPU there's a need to substitute it with 'cuda' so it will be accepted by PyTorch
        if device.upper() == 'GPU':
            device = torch.device('cuda') if torch.cuda.is_available() else log.info('CUDA device is unavailable')
        else:
            device = torch.device(device.lower())
        log.info(f'Torch device was set to: {device}')

        pipe.to(device)
    else:
        raise RuntimeError('==Failure ==: no device to load')

    def get_model_size(models):
        total_size = 0
        for model in models:
            param_size = 0
            for param in model.parameters():
                param_size += param.nelement() * param.element_size()
            buffer_size = 0
            for buffer in model.buffers():
                buffer_size += buffer.nelement() * buffer.element_size()
            model_size_mb = (param_size + buffer_size) / 1024**2

            total_size += model_size_mb
        return total_size
    def get_model_speed(model, inp):
        model(**inp)
        start_time = time.time()
        for i in range(50):
            model(**inp)
        end_time = time.time() - start_time
        return end_time / 50

    if kwargs['torch_compile_backend']:
        backend = kwargs['torch_compile_backend']
        quantize = kwargs["quantize"]
        if(quantize):
            with open('./calibration_data', 'rb') as f:
                unet_calibration_data = pickle.load(f)
            model_dict, configs_dict = export_models(pipe)
            text_encoder = model_dict['text_encoder']
            text_encoder_2 = model_dict['text_encoder_2']
            transformer = model_dict['transformer']
            vae = model_dict['vae']
            with disable_patching():
                with torch.no_grad():
                    text_encoder = nncf.compress_weights(text_encoder)
                    text_encoder_2 = nncf.compress_weights(text_encoder_2)
                    quantized_transformer = nncf.quantize(
                        model=transformer,
                        calibration_dataset=nncf.Dataset(unet_calibration_data[:1]),
                        subset_size=len(unet_calibration_data[:1]),
                        model_type=nncf.ModelType.TRANSFORMER,
                        ignored_scope=nncf.IgnoredScope(names=["conv2d"]),
                        advanced_parameters=nncf.AdvancedQuantizationParameters(
                            weights_range_estimator_params=RangeEstimatorParametersSet.MINMAX,
                            activations_range_estimator_params=RangeEstimatorParametersSet.MINMAX,
                            smooth_quant_alpha=0.7,
                        )
                    )
            # 
            optimized_models_dict = {}        
            optimized_models_dict["transformer"] = run_torch_compile(quantized_transformer, backend)    
            vae.decode = run_torch_compile(vae.decode, backend)
            optimized_models_dict["vae"] = vae
            optimized_models_dict["text_encoder"] = run_torch_compile(text_encoder, backend)
            optimized_models_dict["text_encoder_2"] = run_torch_compile(text_encoder_2, backend)
            pipe = init_pipeline(optimized_models_dict, configs_dict=configs_dict)
        else:
            pipe.transformer = run_torch_compile(pipe.transformer, backend)
            pipe.vae.decode = run_torch_compile(pipe.vae.decode, backend)
            pipe.text_encoder = run_torch_compile(pipe.text_encoder, backend)
            pipe.text_encoder_2 = run_torch_compile(pipe.text_encoder_2, backend)

        print(f'Transformer Size: {get_model_size([pipe.transformer])}')
        print(f'Pipeline Size: {get_model_size([pipe.transformer, pipe.text_encoder, pipe.text_encoder_2, pipe.vae.decoder, pipe.vae.encoder])}')
    return pipe, from_pretrain_time, False, None

def create_ldm_super_resolution_model(model_path, device, **kwargs):
    model_path = Path(model_path)
    from_pretrain_time = 0
    if model_path.exists():
        if model_path.is_dir() and len(os.listdir(model_path)) != 0:
            log.info(f'Load image model from model path:{model_path}')
            model_type = DEFAULT_MODEL_CLASSES[kwargs['use_case']]
            model_class = PT_MODEL_CLASSES_MAPPING[model_type]
            start = time.perf_counter()
            pipe = model_class.from_pretrained(model_path)
            end = time.perf_counter()
            from_pretrain_time = end - start
        else:
            raise RuntimeError(f'==Failure ==: model path:{model_path} is not directory or directory is empty')
    else:
        raise RuntimeError(f'==Failure ==: model path:{model_path} is not exist')

    log.info(f'Model path:{model_path}, from pretrained time: {from_pretrain_time:.2f}s')

    if device:
        # If the device is set to GPU there's a need to substitute it with 'cuda' so it will be accepted by PyTorch
        if device.upper() == 'GPU':
            device = torch.device('cuda') if torch.cuda.is_available() else log.info('CUDA device is unavailable')
        else:
            device = torch.device(device.lower())
        log.info(f'Torch device was set to: {device}')

        pipe.to(device)
    else:
        raise RuntimeError('==Failure ==: no device to load')

    if kwargs['torch_compile_backend']:
        backend = kwargs['torch_compile_backend']
        compiled_model = run_torch_compile(pipe, backend)
        pipe = compiled_model
    return pipe, from_pretrain_time
