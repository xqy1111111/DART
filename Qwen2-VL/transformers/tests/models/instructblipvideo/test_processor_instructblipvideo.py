# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import shutil
import tempfile
import unittest

import numpy as np
import pytest

from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_vision_available():
    from PIL import Image

    from transformers import (
        AutoProcessor,
        BertTokenizerFast,
        GPT2Tokenizer,
        InstructBlipVideoImageProcessor,
        InstructBlipVideoProcessor,
        PreTrainedTokenizerFast,
    )


@require_vision
# Copied from tests.models.instructblip.test_processor_instructblip.InstructBlipProcessorTest with InstructBlip->InstructBlipVideo, BlipImageProcessor->InstructBlipVideoImageProcessor
class InstructBlipVideoProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = InstructBlipVideoProcessor

    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()

        image_processor = InstructBlipVideoImageProcessor()
        tokenizer = GPT2Tokenizer.from_pretrained("hf-internal-testing/tiny-random-GPT2Model")
        qformer_tokenizer = BertTokenizerFast.from_pretrained("hf-internal-testing/tiny-random-bert")

        processor = InstructBlipVideoProcessor(image_processor, tokenizer, qformer_tokenizer)

        processor.save_pretrained(self.tmpdirname)

    def get_tokenizer(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).tokenizer

    def get_image_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).image_processor

    def get_qformer_tokenizer(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).qformer_tokenizer

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    # Ignore copy
    def prepare_image_inputs(self):
        """This function prepares a list of list of PIL images"""

        video_inputs = [
            [Image.fromarray(np.random.randint(255, size=(30, 400, 3), dtype=np.uint8)) for _ in range(5)]
            for _ in range(2)
        ]
        return video_inputs

    def test_save_load_pretrained_additional_features(self):
        processor = InstructBlipVideoProcessor(
            tokenizer=self.get_tokenizer(),
            image_processor=self.get_image_processor(),
            qformer_tokenizer=self.get_qformer_tokenizer(),
        )
        processor.save_pretrained(self.tmpdirname)

        tokenizer_add_kwargs = self.get_tokenizer(bos_token="(BOS)", eos_token="(EOS)")
        image_processor_add_kwargs = self.get_image_processor(do_normalize=False, padding_value=1.0)

        processor = InstructBlipVideoProcessor.from_pretrained(
            self.tmpdirname, bos_token="(BOS)", eos_token="(EOS)", do_normalize=False, padding_value=1.0
        )

        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer_add_kwargs.get_vocab())
        self.assertIsInstance(processor.tokenizer, PreTrainedTokenizerFast)

        self.assertEqual(processor.image_processor.to_json_string(), image_processor_add_kwargs.to_json_string())
        self.assertIsInstance(processor.image_processor, InstructBlipVideoImageProcessor)
        self.assertIsInstance(processor.qformer_tokenizer, BertTokenizerFast)

    def test_image_processor(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()
        qformer_tokenizer = self.get_qformer_tokenizer()

        processor = InstructBlipVideoProcessor(
            tokenizer=tokenizer, image_processor=image_processor, qformer_tokenizer=qformer_tokenizer
        )

        image_input = self.prepare_image_inputs()

        input_feat_extract = image_processor(image_input, return_tensors="np")
        input_processor = processor(images=image_input, return_tensors="np")

        for key in input_feat_extract.keys():
            self.assertAlmostEqual(input_feat_extract[key].sum(), input_processor[key].sum(), delta=1e-2)

    def test_tokenizer(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()
        qformer_tokenizer = self.get_qformer_tokenizer()

        processor = InstructBlipVideoProcessor(
            tokenizer=tokenizer, image_processor=image_processor, qformer_tokenizer=qformer_tokenizer
        )

        input_str = ["lower newer"]

        encoded_processor = processor(text=input_str)

        encoded_tokens = tokenizer(input_str, return_token_type_ids=False)
        encoded_tokens_qformer = qformer_tokenizer(input_str, return_token_type_ids=False)

        for key in encoded_tokens.keys():
            self.assertListEqual(encoded_tokens[key], encoded_processor[key])

        for key in encoded_tokens_qformer.keys():
            self.assertListEqual(encoded_tokens_qformer[key], encoded_processor["qformer_" + key])

    def test_processor(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()
        qformer_tokenizer = self.get_qformer_tokenizer()

        processor = InstructBlipVideoProcessor(
            tokenizer=tokenizer, image_processor=image_processor, qformer_tokenizer=qformer_tokenizer
        )

        input_str = "lower newer"
        image_input = self.prepare_image_inputs()

        inputs = processor(text=input_str, images=image_input)

        self.assertListEqual(
            list(inputs.keys()),
            ["input_ids", "attention_mask", "qformer_input_ids", "qformer_attention_mask", "pixel_values"],
        )

        # test if it raises when no input is passed
        with pytest.raises(ValueError):
            processor()

    def test_tokenizer_decode(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()
        qformer_tokenizer = self.get_qformer_tokenizer()

        processor = InstructBlipVideoProcessor(
            tokenizer=tokenizer, image_processor=image_processor, qformer_tokenizer=qformer_tokenizer
        )

        predicted_ids = [[1, 4, 5, 8, 1, 0, 8], [3, 4, 3, 1, 1, 8, 9]]

        decoded_processor = processor.batch_decode(predicted_ids)
        decoded_tok = tokenizer.batch_decode(predicted_ids)

        self.assertListEqual(decoded_tok, decoded_processor)

    def test_model_input_names(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()
        qformer_tokenizer = self.get_qformer_tokenizer()

        processor = InstructBlipVideoProcessor(
            tokenizer=tokenizer, image_processor=image_processor, qformer_tokenizer=qformer_tokenizer
        )

        input_str = "lower newer"
        image_input = self.prepare_image_inputs()

        inputs = processor(text=input_str, images=image_input)

        self.assertListEqual(
            list(inputs.keys()),
            ["input_ids", "attention_mask", "qformer_input_ids", "qformer_attention_mask", "pixel_values"],
        )

    # Override as InstructBlipVideoProcessor has qformer_tokenizer
    @require_vision
    @require_torch
    def test_tokenizer_defaults_preserved_by_kwargs(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        image_processor = self.get_component("image_processor")
        tokenizer = self.get_component("tokenizer", max_length=117, padding="max_length")
        qformer_tokenizer = self.get_component("qformer_tokenizer", max_length=117, padding="max_length")

        processor = self.processor_class(
            tokenizer=tokenizer, image_processor=image_processor, qformer_tokenizer=qformer_tokenizer
        )
        self.skip_processor_without_typed_kwargs(processor)
        input_str = "lower newer"
        image_input = self.prepare_image_inputs()

        inputs = processor(text=input_str, images=image_input, return_tensors="pt")
        self.assertEqual(len(inputs["input_ids"][0]), 117)

    # Override as InstructBlipVideoProcessor has qformer_tokenizer
    @require_torch
    @require_vision
    def test_image_processor_defaults_preserved_by_image_kwargs(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        image_processor = self.get_component("image_processor", size=(234, 234))
        tokenizer = self.get_component("tokenizer", max_length=117, padding="max_length")
        qformer_tokenizer = self.get_component("qformer_tokenizer", max_length=117, padding="max_length")

        processor = self.processor_class(
            tokenizer=tokenizer, image_processor=image_processor, qformer_tokenizer=qformer_tokenizer
        )
        self.skip_processor_without_typed_kwargs(processor)

        input_str = "lower newer"
        image_input = self.prepare_image_inputs()

        inputs = processor(text=input_str, images=image_input)
        self.assertEqual(len(inputs["pixel_values"][0][0]), 234)

    # Override as InstructBlipVideoProcessor has qformer_tokenizer
    @require_vision
    @require_torch
    def test_kwargs_overrides_default_tokenizer_kwargs(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        image_processor = self.get_component("image_processor")
        tokenizer = self.get_component("tokenizer", padding="longest")
        qformer_tokenizer = self.get_component("qformer_tokenizer", padding="longest")

        processor = self.processor_class(
            tokenizer=tokenizer, image_processor=image_processor, qformer_tokenizer=qformer_tokenizer
        )
        self.skip_processor_without_typed_kwargs(processor)
        input_str = "lower newer"
        image_input = self.prepare_image_inputs()

        inputs = processor(
            text=input_str, images=image_input, return_tensors="pt", max_length=112, padding="max_length"
        )
        self.assertEqual(len(inputs["input_ids"][0]), 112)

    # Override as InstructBlipVideoProcessor has qformer_tokenizer
    @require_torch
    @require_vision
    def test_kwargs_overrides_default_image_processor_kwargs(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        image_processor = self.get_component("image_processor", size=(234, 234))
        tokenizer = self.get_component("tokenizer", max_length=117, padding="max_length")
        qformer_tokenizer = self.get_component("qformer_tokenizer", max_length=117, padding="max_length")

        processor = self.processor_class(
            tokenizer=tokenizer, image_processor=image_processor, qformer_tokenizer=qformer_tokenizer
        )
        self.skip_processor_without_typed_kwargs(processor)

        input_str = "lower newer"
        image_input = self.prepare_image_inputs()

        inputs = processor(text=input_str, images=image_input, size=[224, 224])
        self.assertEqual(len(inputs["pixel_values"][0][0]), 224)

    # Override as InstructBlipVideoProcessor has qformer_tokenizer
    @require_torch
    @require_vision
    def test_unstructured_kwargs(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        image_processor = self.get_component("image_processor")
        tokenizer = self.get_component("tokenizer")
        qformer_tokenizer = self.get_component("qformer_tokenizer")

        processor = self.processor_class(
            tokenizer=tokenizer, image_processor=image_processor, qformer_tokenizer=qformer_tokenizer
        )
        self.skip_processor_without_typed_kwargs(processor)

        input_str = "lower newer"
        image_input = self.prepare_image_inputs()
        inputs = processor(
            text=input_str,
            images=image_input,
            return_tensors="pt",
            size={"height": 214, "width": 214},
            padding="max_length",
            max_length=76,
        )

        self.assertEqual(inputs["pixel_values"].shape[2], 214)
        self.assertEqual(len(inputs["input_ids"][0]), 76)

    # Override as InstructBlipVideoProcessor has qformer_tokenizer
    @require_torch
    @require_vision
    def test_unstructured_kwargs_batched(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        image_processor = self.get_component("image_processor")
        tokenizer = self.get_component("tokenizer")
        qformer_tokenizer = self.get_component("qformer_tokenizer")

        processor = self.processor_class(
            tokenizer=tokenizer, image_processor=image_processor, qformer_tokenizer=qformer_tokenizer
        )
        self.skip_processor_without_typed_kwargs(processor)

        input_str = ["lower newer", "upper older longer string"]
        image_input = self.prepare_image_inputs() * 2
        inputs = processor(
            text=input_str,
            images=image_input,
            return_tensors="pt",
            size={"height": 214, "width": 214},
            padding="longest",
            max_length=76,
        )

        self.assertEqual(inputs["pixel_values"].shape[2], 214)

        self.assertEqual(len(inputs["input_ids"][0]), 6)

    # Override as InstructBlipVideoProcessor has qformer_tokenizer
    @require_torch
    @require_vision
    def test_doubly_passed_kwargs(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        image_processor = self.get_component("image_processor")
        tokenizer = self.get_component("tokenizer")
        qformer_tokenizer = self.get_component("qformer_tokenizer")

        processor = self.processor_class(
            tokenizer=tokenizer, image_processor=image_processor, qformer_tokenizer=qformer_tokenizer
        )
        self.skip_processor_without_typed_kwargs(processor)

        input_str = ["lower newer"]
        image_input = self.prepare_image_inputs()
        with self.assertRaises(ValueError):
            _ = processor(
                text=input_str,
                images=image_input,
                images_kwargs={"size": {"height": 222, "width": 222}},
                size={"height": 214, "width": 214},
            )

    # Override as InstructBlipVideoProcessor has qformer_tokenizer
    @require_torch
    @require_vision
    def test_structured_kwargs_nested(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        image_processor = self.get_component("image_processor")
        tokenizer = self.get_component("tokenizer")
        qformer_tokenizer = self.get_component("qformer_tokenizer")

        processor = self.processor_class(
            tokenizer=tokenizer, image_processor=image_processor, qformer_tokenizer=qformer_tokenizer
        )
        self.skip_processor_without_typed_kwargs(processor)

        input_str = "lower newer"
        image_input = self.prepare_image_inputs()

        # Define the kwargs for each modality
        all_kwargs = {
            "common_kwargs": {"return_tensors": "pt"},
            "images_kwargs": {"size": {"height": 214, "width": 214}},
            "text_kwargs": {"padding": "max_length", "max_length": 76},
        }

        inputs = processor(text=input_str, images=image_input, **all_kwargs)
        self.skip_processor_without_typed_kwargs(processor)

        self.assertEqual(inputs["pixel_values"].shape[2], 214)

        self.assertEqual(len(inputs["input_ids"][0]), 76)

    # Override as InstructBlipVideoProcessor has qformer_tokenizer
    @require_torch
    @require_vision
    def test_structured_kwargs_nested_from_dict(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")

        image_processor = self.get_component("image_processor")
        tokenizer = self.get_component("tokenizer")
        qformer_tokenizer = self.get_component("qformer_tokenizer")

        processor = self.processor_class(
            tokenizer=tokenizer, image_processor=image_processor, qformer_tokenizer=qformer_tokenizer
        )
        self.skip_processor_without_typed_kwargs(processor)
        input_str = "lower newer"
        image_input = self.prepare_image_inputs()

        # Define the kwargs for each modality
        all_kwargs = {
            "common_kwargs": {"return_tensors": "pt"},
            "images_kwargs": {"size": {"height": 214, "width": 214}},
            "text_kwargs": {"padding": "max_length", "max_length": 76},
        }

        inputs = processor(text=input_str, images=image_input, **all_kwargs)
        self.assertEqual(inputs["pixel_values"].shape[2], 214)

        self.assertEqual(len(inputs["input_ids"][0]), 76)
