# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import numpy as np
from transformers import AutoTokenizer
from typing import Dict, Tuple, List
import openvino_genai
import json

from ov_genai_test_utils import (
    get_models_list,
    get_chat_models_list,
    read_model,
    model_tmp_path
)


def load_genai_tokenizer_with_configs(configs: List[Tuple], temp_path):
    for json_file in temp_path.glob("*.json"):
        json_file.unlink()

    for config_json, config_name in configs:
        with (temp_path / config_name).open('w') as f:
            json.dump(config_json, f)

    ov_tokenizer = openvino_genai.Tokenizer(temp_path)

    for _, config_name in configs:
        os.remove(temp_path / config_name)

    return ov_tokenizer


def get_chat_templates():
    # Returns chat templates saved in tokenizer_configs.py, 
    # but skips some models that currently are not processed correctly.

    skipped_models = {
        # TODO: openchat/openchat_3.5 and berkeley-nest/Starling-LM-7B-alpha have the same template.
        # Need to enable and unskip, since it's preset in continuous batching and has >100 000 downloads.
        "openchat/openchat-3.5-0106",
        
        # These models fail even on HF so no need to check if applying chat matches.
        "vibhorag101/llama-2-13b-chat-hf-phr_mental_therapy",
        "codellama/CodeLlama-34b-Instruct-hf",
        "deepseek-ai/deepseek-math-7b-rl",
        "allenai/tulu-2-7b",
        "alexsobolev/IcaroLM",
        "tokyotech-llm/Swallow-7b-instruct-v0.1",
        "bofenghuang/vigogne-2-7b-chat",
        "OpenBuddy/openbuddy-mistral2-7b-v20.3-32k",
        "AliAbdelrasheed/maqa_llama_4bit",
        "stephenlzc/Mistral-7B-v0.3-Chinese-Chat-uncensored",

        # TODO: Need to support chat templates in more models: CVS-145963
        # Either ov_genai is unable to parse chat_template or results do not match with HF.
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "databricks/dbrx-instruct", # Chat template is not supported by Jinja2Cpp
        "mosaicml/mpt-30b-chat",
        "deepseek-ai/deepseek-coder-6.7b-instruct", # Chat template is not supported by Jinja2Cpp
        "maldv/winter-garden-7b-alpha", # Chat template is not supported by Jinja2Cpp
        "ishorn5/RTLCoder-Deepseek-v1.1", # Chat template is not supported by Jinja2Cpp
        "openchat/openchat-3.5-0106",
        "casperhansen/llama-3-70b-instruct-awq",
        "TheBloke/deepseek-coder-33B-instruct-GPTQ",
        "AI-Sweden-Models/gpt-sw3-356m-instruct",
        "google/gemma-7b-it",
        "THUDM/cogvlm2-llama3-chat-19B",
        "KnutJaegersberg/internlm-20b-llama",
        "maywell/Synatra-Mixtral-8x7B",
        "MediaTek-Research/Breeze-7B-Instruct-v1_0",
        "bofenghuang/vigostral-7b-chat",
        "meetkai/functionary-small-v2.5", # Chat template is not supported by Jinja2Cpp
        "openchat/openchat-3.6-8b-20240522",
        "tenyx/TenyxChat-7B-v1",
        "LoneStriker/TinyLlama-1.1B-32k-Instruct-3.0bpw-h6-exl2",
        "yam-peleg/Hebrew-Gemma-11B-V2",
        "shenzhi-wang/Llama3-8B-Chinese-Chat", # AssertionError
        "nlpai-lab/KULLM3",
        "HuggingFaceH4/zephyr-7b-gemma-sft-v0.1",
        "MediaTek-Research/Breeze-7B-Instruct-v0_1", 
        "shanchen/llama3-8B-slerp-biomed-chat-chinese", # AssertionError
        "MLP-KTLim/llama-3-Korean-Bllossom-8B",
        "aloobun/CosmicBun-8B", # Chat template is not supported by Jinja2Cpp
        "codellama/CodeLlama-70b-Instruct-hf",
        "gorilla-llm/gorilla-openfunctions-v2", # Chat template is not supported by Jinja2Cpp
        "BramVanroy/Llama-2-13b-chat-dutch"
    }

    from tokenizer_configs import get_tokenizer_configs
    return [(k, v) for k, v in get_tokenizer_configs().items() if k not in skipped_models]


prompts = [
    'table is made of',
    '你好！ 你好嗎？',
    'Alan Turing was a',
    'The Sun is yellow because',
    ['The Sun is yellow because', 'Alan Turing was a', 'Alan Turing was a']
]
@pytest.mark.parametrize("model_descr", get_models_list())
@pytest.mark.parametrize("prompt", prompts)
@pytest.mark.precommit
@pytest.mark.nightly
def test_encode(model_descr, prompt):
    model_id, path, hf_tokenizer, opt_model, ov_pipe = read_model(model_descr)
    ov_tokenizer = ov_pipe.get_tokenizer()

    encoded_ov = ov_tokenizer.encode(prompt).input_ids.data
    if isinstance(prompt, list):
        encoded_hf = hf_tokenizer.batch_encode_plus(prompt)['input_ids']
        for tokens_ov, tokens_hf in zip(encoded_ov, encoded_hf):
            assert np.all(tokens_ov == tokens_hf)
    else:
        encoded_hf = hf_tokenizer.encode(prompt)
        assert np.all(encoded_hf == encoded_ov[0])


encoded_prompts = [
    [1, 1591, 338, 1754, 310],
    [1, 17102,   323,  3864,   471,   263],

    # chineze characters
    [1, 29871, 30919, 31076, 30584, 29871, 30919, 31076, 232, 154, 145, 30882],

    # On meta-llama/Meta-Llama-3-8B-Instruct this becomes longer  after removing the last token
    [3113, 264, 364, 267],

    # batched tokens
    [[1, 1591, 338, 1754, 310], [1, 1591, 338, 1754, 310], [1, 17102,   323,  3864,   471,   263]]
]
@pytest.mark.parametrize("model_descr", get_models_list())
@pytest.mark.parametrize("encoded_prompt", encoded_prompts)
@pytest.mark.precommit
def test_decode(model_descr, encoded_prompt):
    model_id, path, hf_tokenizer, opt_model, ov_pipe = read_model(model_descr)
    ov_tokenizer = ov_pipe.get_tokenizer()
    decoded_ov = ov_tokenizer.decode(encoded_prompt)

    if isinstance(encoded_prompt[0], list):
        decoded_hf = hf_tokenizer.batch_decode(encoded_prompt, skip_special_tokens=True)
        for tokens_ov, tokens_hf in zip(decoded_ov, decoded_hf):
            assert np.all(tokens_ov == tokens_hf)
    else:
        decoded_hf = hf_tokenizer.decode(encoded_prompt, skip_special_tokens=True)
        assert decoded_hf == decoded_ov


conversation = [
    {'role': 'user', 'content': '1+1='},
    {'role': 'assistant', 'content': '1 + 1 = 2'},
    {'role': 'user', 'content': 'What is the previous answer?'},
    {'role': 'assistant', 'content': 'The previous answer was: 1 + 1 = 2. Please ask me your next question.'},
    {'role': 'user', 'content': 'Why is the sun yellow?'},
    {'role': 'assistant', 'content': 'Because it emits yeloow light.'},
    {'role': 'user', 'content': 'What was my first question?'},
]
@pytest.mark.precommit
@pytest.mark.nightly
@pytest.mark.parametrize('chat_config', get_chat_templates())
def test_apply_chat_template(model_tmp_path, chat_config: Tuple[str, Dict]):
    tokenizer_config = chat_config[1]

    # Will load openvino_model for tiny-random-phi as a placeholder
    # but indeed only Tokenizer and apply_chat_template will be tested.
    model_id, path, hf_tokenizer, opt_model, ov_pipe = read_model(get_models_list()[0])

    hf_full_history_str = hf_tokenizer.apply_chat_template(conversation,
        add_generation_prompt=False,
        tokenize=False,
        **tokenizer_config)

    ov_tokenizer = load_genai_tokenizer_with_configs([(tokenizer_config, "tokenizer_config.json")], model_tmp_path[1])
    ov_tokenizer.set_chat_template(tokenizer_config['chat_template'])
    ov_full_history_str = ov_tokenizer.apply_chat_template(conversation, add_generation_prompt=False)

    if ov_full_history_str != hf_full_history_str:
        print(f'hf reference: {hf_full_history_str}')
        print(f'ov_genai out: {ov_full_history_str}')
    assert ov_full_history_str == hf_full_history_str


@pytest.mark.precommit
@pytest.mark.nightly
def test_set_chat_template():
    model_descr = get_chat_models_list()[0]
    model_id, path, hf_tokenizer, opt_model, ov_pipe = read_model((model_descr[0], model_descr[1] / '_test_chat'))

    prompt = "how are you?"
    dummy_conversation = [
        {'role': 'user', 'content': prompt},
    ]

    ov_tokenizer = ov_pipe.get_tokenizer()
    identity_chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"

    templated_prompt_inline = ov_tokenizer.apply_chat_template(dummy_conversation, add_generation_prompt=False, chat_template=identity_chat_template)

    ov_tokenizer.set_chat_template(identity_chat_template)
    templated_prompt = ov_tokenizer.apply_chat_template(dummy_conversation, add_generation_prompt=False)

    assert templated_prompt_inline == templated_prompt
    assert prompt == templated_prompt


prompts = [
    '1+1=',
    'What is the previous answer?',
    'Why is the Sun yellow?',
    'What was my first question?',
    ['Why is the Sun yellow?'],
    "若我有一亿美元，在人工智能盛行的今天，我怎样投资才能收益最大化？",
    "מחרוזת בדיקה",
    "Multiline\nstring!\nWow!",
]
@pytest.mark.precommit
@pytest.mark.nightly
@pytest.mark.parametrize("add_special_tokens", [True, False])
@pytest.mark.parametrize("skip_special_tokens", [True, False])
@pytest.mark.parametrize("prompt", prompts)
def test_encode_decode_with_special_tokens_option(add_special_tokens, skip_special_tokens, prompt):
    import numpy as np
    model_descr = get_chat_models_list()[0]
    model_id, path, hf_tokenizer, model_opt, ov_pipe = read_model((model_descr[0], model_descr[1] / '_test_chat'))
    ov_tokenzier = ov_pipe.get_tokenizer()

    # Calling encode with 'add_special_tokens' will set state flag.
    ov_res = ov_tokenzier.encode(prompt, add_special_tokens=add_special_tokens).input_ids.data
    hf_res = hf_tokenizer(prompt, return_tensors="np", add_special_tokens=add_special_tokens)["input_ids"]
    assert np.all(ov_res == hf_res)

    # Decode with 'skip_special_tokens'
    decoded_genai = ov_tokenzier.decode(ov_res, skip_special_tokens=skip_special_tokens)[0]
    decoded_hf = hf_tokenizer.decode(hf_res[0], skip_special_tokens=skip_special_tokens)
    assert decoded_genai == decoded_hf


@pytest.mark.precommit
@pytest.mark.nightly
def test_load_special_tokens_from_config_json(model_tmp_path):
    # test when there is an available config.json
    config_json = {
        "pad_token_id": 422,
        "bos_token_id": 42,
        "eos_token_id": 37,
    }
    tok = load_genai_tokenizer_with_configs([(config_json, "config.json")], model_tmp_path[1])
    assert tok.get_pad_token_id() == config_json['pad_token_id']
    assert tok.get_bos_token_id() == config_json['bos_token_id']
    assert tok.get_eos_token_id() == config_json['eos_token_id']


@pytest.mark.precommit
@pytest.mark.nightly
def test_load_special_tokens_from_special_tokens_map_json(model_tmp_path):
    # test with special_tokens_map
    special_tokens_map_json = {
        "pad_token": {"content": "<custom_pad>"},
        "bos_token": {"content": "<custom_bos>"},
        "eos_token": {"content": "<custom_eos>"},
    }
    tok = load_genai_tokenizer_with_configs([(special_tokens_map_json, "special_tokens_map.json")], model_tmp_path[1])
    assert tok.get_pad_token() == special_tokens_map_json['pad_token']["content"]
    assert tok.get_bos_token() == special_tokens_map_json['bos_token']["content"]
    assert tok.get_eos_token() == special_tokens_map_json['eos_token']["content"]


@pytest.mark.precommit
@pytest.mark.nightly
@pytest.mark.skip(reason="CVS-158682 - RTInfo is not modified in tests for unknown reasons")
def test_load_special_tokens_from_tokenizer_config_json(model_tokenizers_tmp_path):
    # special_tokens_map is not available
    # but tokenize_config.json exists
    # will load both string and integer representations
    tok_config_json = {
        "added_tokens_decoder": {
            "422": {"content": "<pad>"},
            "37": {"content": "<s>"},
            "42": {"content": "</s>"},
        },
        "pad_token": "<pad>",
        "bos_token": "<s>",
        "eos_token": "</s>",
    }

    tok = load_genai_tokenizer_with_configs([(tok_config_json, "tokenizer_config.json")], model_tokenizers_tmp_path[1])
    assert tok.get_pad_token() == tok_config_json['pad_token']
    assert tok.get_bos_token() == tok_config_json['bos_token']
    assert tok.get_eos_token() == tok_config_json['eos_token']

    assert tok.get_pad_token_id() == 422
    assert tok.get_bos_token_id() == 37
    assert tok.get_eos_token_id() == 42


@pytest.mark.precommit
@pytest.mark.nightly
def test_load_special_tokens_from_tokenizer_config_and_config_json(model_tmp_path):
    # both config.json is available and tokenizer_config.json available
    # check that it does not read int values from tokenizer_config.json if they are in config.json
    tok_config_json = {
    "added_tokens_decoder": {
        # integers differ from config.json to check they don't override config.json
        "777": {"content": "<pad>"},
        "888": {"content": "<s>"},
        "656": {"content": "</s>"},
    },
    "pad_token": "<pad>",
    "bos_token": "<s>",
    "eos_token": "</s>",
    }
    config_json = {
        "pad_token_id": 422,
        "bos_token_id": 42,
        "eos_token_id": 37,
    }
    configs = [
        (tok_config_json, "tokenizer_config.json"),
        (config_json, "config.json")
    ]
    tok = load_genai_tokenizer_with_configs(configs, model_tmp_path[1])
    assert tok.get_pad_token_id() == config_json['pad_token_id']
    assert tok.get_bos_token_id() == config_json['bos_token_id']
    assert tok.get_eos_token_id() == config_json['eos_token_id']

    assert tok.get_pad_token() == tok_config_json['pad_token']
    assert tok.get_bos_token() == tok_config_json['bos_token']
    assert tok.get_eos_token() == tok_config_json['eos_token']


@pytest.mark.precommit
@pytest.mark.nightly
@pytest.mark.xfail(
    raises=AssertionError,
    reason="CVS-143410 ov tokenizer should be aligned with hf",
    strict=False,
)
def test_load_special_tokens_from_special_tokens_map_json_with_string_repr(model_tmp_path):
    # only string representation is provided, find token integers by inference
    model_id, temp_path = model_tmp_path
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    special_tokens_map_json = {}
    token_str_int_map = {}
    special_token_names = ['pad_token', 'bos_token', 'eos_token']
    for token_str in special_token_names:
        if hasattr(tokenizer, token_str):
            token_val = getattr(tokenizer, token_str)
            special_tokens_map_json.update({token_str: {"content": token_val}})
            token_id = tokenizer(token_val, add_special_tokens=False)['input_ids'][0]
            token_str_int_map.update({token_str: token_id})

    # since only string representations are present in the json will try to get by inference
    tok = load_genai_tokenizer_with_configs([(special_tokens_map_json, "special_tokens_map.json")], temp_path)

    # check ids inferred correctly for special tokens existing if HF tokenizer
    if 'pad_token' in token_str_int_map:
        assert tok.get_pad_token_id() == token_str_int_map['pad_token']
    if 'bos_token' in token_str_int_map:
        assert tok.get_bos_token_id() == token_str_int_map['bos_token']
    if 'eos_token' in token_str_int_map:
        assert tok.get_eos_token_id() == token_str_int_map['eos_token']

