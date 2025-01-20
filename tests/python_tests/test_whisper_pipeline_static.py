# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from test_whisper_pipeline import get_whisper_models_list, get_samples_from_dataset
from transformers import WhisperProcessor, AutoTokenizer
from optimum.intel.openvino import OVModelForSpeechSeq2Seq
import openvino_genai as ov_genai
import openvino_tokenizers
import openvino
import pytest
import pathlib

# This test suite is designed specifically to validate the functionality 
# and robustness of the WhisperStaticPipeline on NPUW:CPU.
config = {"NPU_USE_NPUW" : "YES",
          "NPUW_DEVICES" : "CPU",
          "NPUW_ONLINE_PIPELINE" : "NONE"}

def load_and_save_whisper_model(params, stateful=False, **tokenizer_kwargs):
    model_id, path = params

    processor = WhisperProcessor.from_pretrained(model_id, trust_remote_code=True)
    if not stateful:
        path = pathlib.Path(f"{path}_with_past")

    if not (path / "openvino_encoder_model.xml").exists():
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        ov_tokenizer, ov_detokenizer = openvino_tokenizers.convert_tokenizer(
            tokenizer,
            with_detokenizer=True,
            clean_up_tokenization_spaces=False,
            **tokenizer_kwargs,
        )

        openvino.save_model(ov_tokenizer, path / "openvino_tokenizer.xml")
        openvino.save_model(ov_detokenizer, path / "openvino_detokenizer.xml")

        # to store tokenizer config jsons with special tokens
        tokenizer.save_pretrained(path)

        opt_model = OVModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            export=True,
            trust_remote_code=True,
            stateful=stateful,
            compile=False,
            device="CPU",
            load_in_8bit=False,
        )
        opt_model.generation_config.save_pretrained(path)
        opt_model.config.save_pretrained(path)
        opt_model.save_pretrained(path)
        processor.save_pretrained(path)
    
    return model_id, path

def get_results_cpu_npu(model_path, audio_sample, **config_kwargs):
    cpu_pipe = ov_genai.WhisperPipeline(model_path, "CPU")
    expected = cpu_pipe.generate(audio_sample, **config_kwargs)

    npu_pipe = ov_genai.WhisperPipeline(model_path, "NPU", **config)
    actual_out = npu_pipe.generate(audio_sample, **config_kwargs)

    return expected, actual_out

def compare_results_with_assert(expected, actual_out):
    assert len(expected.texts) == len(actual_out.texts)

    for i in range(0, len(expected.texts)):
        if expected.texts[i] != actual_out.texts[i]:
            print(f'expected: {expected.texts[i]}\n')
            print(f'actual_out: {actual_out.texts[i]}')
        assert expected.texts[i] == actual_out.texts[i]


@pytest.mark.parametrize("model_descr", get_whisper_models_list(tiny_only=True))
@pytest.mark.parametrize("test_sample", get_samples_from_dataset(language="en", length=1))
@pytest.mark.precommit
def test_static_whisper_generation_compare_with_cpu(model_descr, test_sample):
    model_id, model_path = load_and_save_whisper_model(model_descr)

    expected, actual_out = get_results_cpu_npu(model_path, test_sample)

    compare_results_with_assert(expected, actual_out)


@pytest.mark.parametrize("model_descr", get_whisper_models_list(tiny_only=True))
@pytest.mark.parametrize("test_sample",
    [
        *get_samples_from_dataset(language="fr", length=2),
        *get_samples_from_dataset(language="de", length=2),
        *get_samples_from_dataset(language="es", length=2),
    ],)
@pytest.mark.precommit
def test_static_whisper_autodetect(model_descr, test_sample):
    model_id, model_path = load_and_save_whisper_model(model_descr)

    expected, actual_out = get_results_cpu_npu(model_path, test_sample)

    compare_results_with_assert(expected, actual_out)


@pytest.mark.parametrize("model_descr", get_whisper_models_list(tiny_only=True))
@pytest.mark.parametrize(
    "test_sample", get_samples_from_dataset(language="de", length=3)
)
@pytest.mark.precommit
def test_static_whisper_language_de(model_descr, test_sample):
    model_id, model_path = load_and_save_whisper_model(model_descr)

    expected, actual_out = get_results_cpu_npu(model_path, test_sample, max_new_tokens=30, language="<|de|>")

    compare_results_with_assert(expected, actual_out)


@pytest.mark.parametrize("model_descr", get_whisper_models_list(tiny_only=True))
@pytest.mark.parametrize(
    "test_sample", get_samples_from_dataset(language="fr", length=3)
)
@pytest.mark.precommit
def test_static_whisper_language_fr(model_descr, test_sample):
    model_id, model_path = load_and_save_whisper_model(model_descr)

    expected, actual_out = get_results_cpu_npu(model_path, test_sample, max_new_tokens=30, language="<|fr|>")

    compare_results_with_assert(expected, actual_out)


@pytest.mark.parametrize("model_descr", get_whisper_models_list(tiny_only=True))
@pytest.mark.parametrize(
    "test_sample", get_samples_from_dataset(language="ru", length=3)
)
@pytest.mark.precommit
def test_static_whisper_language_ru(model_descr, test_sample):
    model_id, model_path = load_and_save_whisper_model(model_descr)

    expected, actual_out = get_results_cpu_npu(model_path, test_sample, max_new_tokens=30, language="<|ru|>")

    compare_results_with_assert(expected, actual_out)


@pytest.mark.parametrize("model_descr", get_whisper_models_list(tiny_only=True))
@pytest.mark.parametrize("test_sample", get_samples_from_dataset(language="en", length=1, long_form=True))
@pytest.mark.precommit
def test_static_whisper_generation_long(model_descr, test_sample):
    model_id, model_path = load_and_save_whisper_model(model_descr)

    expected, actual_out = get_results_cpu_npu(model_path, test_sample)

    compare_results_with_assert(expected, actual_out)
