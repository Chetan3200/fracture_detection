"""CPU-only contract tests; no torch/transformers/model download required."""
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

MODULE = Path(__file__).resolve().parents[1] / "medgemma_model.py"
spec = importlib.util.spec_from_file_location("medgemma_model_under_test", MODULE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def wrap(payload):
    return "Final Answer: ```json\n" + payload + "\n```"


def object_(box=None, **extra):
    return {"box_2d": [10, 20, 30, 40] if box is None else box,
            "label": "fracture", **extra}


class Tokenizer:
    def __init__(self, pieces):
        self.pieces = pieces

    def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        assert skip_special_tokens is False
        assert clean_up_tokenization_spaces is False
        return "".join(self.pieces[i] for i in ids)


class ParseTests(unittest.TestCase):
    def test_empty_is_valid(self):
        self.assertEqual(m.parse_answer(wrap("[]")), ([], []))

    def test_duplicate_coordinates_have_distinct_ordered_spans(self):
        text = wrap(json.dumps([object_(), object_()]))
        boxes, spans = m.parse_answer(text)
        self.assertEqual(len(boxes), 2)
        self.assertLess(spans[0][1], spans[1][0])
        for span in spans:
            self.assertEqual(json.loads(text[slice(*span)]), boxes[0]["box_2d"])

    def test_escaped_keys_and_reversed_members(self):
        text = wrap('[{"label":"fracture","box_\\u0032d":[0, 1e1, 9.5e2, 1000]}]')
        boxes, spans = m.parse_answer(text)
        self.assertEqual(boxes[0]["box_2d"], [0, 10, 950, 1000])
        self.assertEqual(text[slice(*spans[0])], "[0, 1e1, 9.5e2, 1000]")

    def test_duplicate_keys_rejected_including_escapes(self):
        for payload in ('[{"label":"fracture","label":"fracture","box_2d":[0,0,1,1]}]',
                        '[{"label":"fracture","box_2d":[0,0,1,1],"box_\\u0032d":[0,0,1,1]}]'):
            with self.subTest(payload=payload), self.assertRaises(m.SchemaFailure):
                m.parse_answer(wrap(payload))

    def test_invalid_coordinates(self):
        for box in ([True, 0, 2, 2], ["0", 0, 2, 2], [-1, 0, 2, 2],
                    [0, 0, 1001, 2], [0, 2, 1, 2], [2, 0, 1, 2],
                    [0, 0, float("inf"), 2], [0, 0, float("nan"), 2],
                    [0, 0, 1], [0, 0, 1, 1, 1], [0, 0, 10**1000, 1]):
            with self.subTest(box=box), self.assertRaises(m.SchemaFailure):
                m.parse_answer(wrap(json.dumps([object_(box)])))

    def test_nonfinite_numeric_overflow(self):
        with self.assertRaises(m.SchemaFailure):
            m.parse_answer(wrap('[{"box_2d":[0,0,1e999,10],"label":"fracture"}]'))

    def test_wrong_schema(self):
        for value in ({}, [0], [object_(confidence=0.9)],
                      [{"label": "other", "box_2d": [0, 0, 1, 1]}],
                      [{"label": "fracture"}]):
            with self.subTest(value=value), self.assertRaises(m.SchemaFailure):
                m.parse_answer(wrap(json.dumps(value)))

    def test_invalid_json_and_wrapper(self):
        for text in ("[]", wrap("["), wrap("[] []"), wrap("[]") + " explanation",
                     "explanation " + wrap("[]"), wrap('[{"label":"fracture",}]')):
            with self.subTest(text=text), self.assertRaises(m.FormatFailure):
                m.parse_answer(text)

    def test_no_partial_salvage(self):
        with self.assertRaises(m.SchemaFailure):
            m.parse_answer(wrap(json.dumps([object_(), {"label": "other"}])))


class AlignmentTests(unittest.TestCase):
    def test_terminal_eos_only_removed(self):
        tokenizer = Tokenizer({0: "ab", 1: "cd", 2: "<eos>"})
        text, spans = m.decode_with_spans(tokenizer, [0, 1, 2], [2])
        self.assertEqual(text, "abcd")
        self.assertEqual(spans, [(0, 2), (2, 4)])

    def test_internal_special_retained(self):
        tokenizer = Tokenizer({0: "ab", 1: "cd", 2: "<eos>"})
        text, spans = m.decode_with_spans(tokenizer, [0, 2, 1, 2], [2])
        self.assertEqual(text, "ab<eos>cd")
        self.assertEqual(len(spans), 3)

    def test_no_terminal_eos(self):
        self.assertEqual(m.decode_with_spans(Tokenizer({0: "abc"}), [0], [2]),
                         ("abc", [(0, 3)]))

    def test_nonprefix_stability_aborts(self):
        class Unstable:
            def decode(self, ids, **kwargs):
                return {0: "", 1: "a�", 2: "aé"}[len(ids)]
        with self.assertRaisesRegex(RuntimeError, "Non-prefix-stable"):
            m.decode_with_spans(Unstable(), [1, 2], [])

    def test_empty_decode(self):
        self.assertEqual(m.decode_with_spans(Tokenizer({}), [], []), ("", []))

    def test_boundary_overlap_and_original_pixels(self):
        boxes = [object_([100, 200, 500, 800])]
        # First/last selected tokens straddle the coordinate-span boundaries.
        pred, scores = m.score_boxes(boxes, [(3, 8)], [(0, 2), (2, 5), (5, 9), (9, 12)],
                                    [-9, math.log(0.25), math.log(1.0), -9], 200, 100)
        self.assertEqual(pred, [[40, 10, 160, 50]])
        self.assertAlmostEqual(scores[0], 0.5)

    def test_score_errors_abort(self):
        for spans, probabilities in (([(0, 1)], [-1]), ([(0, 9)], [float("nan")]),
                                     ([(0, 9)], [1.0]), ([], [])):
            with self.subTest(spans=spans), self.assertRaises(RuntimeError):
                m.score_boxes([object_()], [(2, 5)], spans, probabilities, 100, 100)

    def test_all_boxes_retained_even_very_low_score(self):
        boxes = [object_(), object_()]
        pred, scores = m.score_boxes(boxes, [(0, 1), (1, 2)], [(0, 1), (1, 2)],
                                    [-100, -200], 100, 100)
        self.assertEqual(len(pred), 2)
        self.assertEqual(scores, [math.exp(-100), math.exp(-200)])


class ParseBeforeAlignmentTests(unittest.TestCase):
    class ByteFallbackTokenizer:
        """The first byte decodes to replacement; the full response replaces it."""
        def __init__(self, full):
            self.full = full
            self.calls = []

        def decode(self, ids, **kwargs):
            self.calls.append(list(ids))
            assert kwargs == {"skip_special_tokens": False, "clean_up_tokenization_spaces": False}
            if len(ids) == 0:
                return ""
            if len(ids) == 1:
                return "�"
            return self.full

    def result(self, tokenizer, cap=8, ids=None):
        return m._prediction_from_tokens(tokenizer, [1, 2, 9] if ids is None else ids,
                                         [9], cap, [-1, -1, -1], 100, 100)

    def test_invalid_unicode_prose_does_not_attempt_alignment(self):
        tokenizer = self.ByteFallbackTokenizer("é This is not the answer format.")
        result = self.result(tokenizer)
        self.assertEqual(result["status"], "format_failure")
        self.assertEqual(result["pred_xyxy"], [])
        self.assertEqual(tokenizer.calls, [[1, 2]])

    def test_valid_empty_does_not_attempt_alignment(self):
        tokenizer = self.ByteFallbackTokenizer("\u00a0" + wrap("[]"))
        result = self.result(tokenizer)
        self.assertEqual(result["status"], "valid_empty")
        self.assertEqual(tokenizer.calls, [[1, 2]])

    def test_invalid_schema_does_not_attempt_alignment(self):
        tokenizer = self.ByteFallbackTokenizer("\u00a0" + wrap('[{"label":"other"}]'))
        self.assertEqual(self.result(tokenizer)["status"], "schema_failure")
        self.assertEqual(tokenizer.calls, [[1, 2]])

    def test_valid_boxes_with_unstable_prefix_still_abort(self):
        tokenizer = self.ByteFallbackTokenizer("\u00a0" + wrap(json.dumps([object_()])))
        with self.assertRaisesRegex(RuntimeError, "Non-prefix-stable"):
            self.result(tokenizer)

    def test_cap_overrides_valid_json_without_alignment(self):
        tokenizer = self.ByteFallbackTokenizer(wrap("[]"))
        result = self.result(tokenizer, cap=2, ids=[1, 2])
        self.assertEqual(result["status"], "format_failure")
        self.assertTrue(result["truncated"])
        self.assertEqual(tokenizer.calls, [[1, 2]])

    def test_eos_at_cap_remains_valid_like_training(self):
        tokenizer = self.ByteFallbackTokenizer(wrap("[]"))
        result = self.result(tokenizer, cap=3)
        self.assertEqual(result["status"], "valid_empty")
        self.assertFalse(result["truncated"])

    def test_valid_boxes_align_and_score(self):
        text = wrap(json.dumps([object_()]))
        tokenizer = Tokenizer({1: text, 9: "<eos>"})
        result = self.result(tokenizer, ids=[1, 9])
        self.assertEqual(result["status"], "valid_nonempty")
        self.assertEqual(result["scores"], [math.exp(-1)])
        self.assertEqual(result["pred_xyxy"], [[2, 1, 4, 3]])

    def test_processor_fallback_order(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            folders = [run / "processor", run / "best_adapter", run / "checkpoint-1"]
            for folder in reversed(folders):
                folder.mkdir()
                for name in ("preprocessor_config.json", "tokenizer_config.json", "tokenizer.json"):
                    (folder / name).write_text("{}")
                self.assertEqual(m._processor_path(run, folders[-1]), folder)


class TransformersRecorderTests(unittest.TestCase):
    def test_actual_pinned_neutral_greedy_generation(self):
        if importlib.util.find_spec("transformers") is None or importlib.util.find_spec("torch") is None:
            self.skipTest("Optional local test requires torch and transformers 4.57.6.")
        import torch
        import transformers
        from transformers import GPT2Config, GPT2LMHeadModel, GenerationConfig, LogitsProcessor
        self.assertEqual(transformers.__version__, "4.57.6", "Run this integration test in the pinned environment.")
        torch.manual_seed(7)
        model = GPT2LMHeadModel(GPT2Config(vocab_size=32, n_positions=32, n_embd=16,
                                         n_layer=1, n_head=2, bos_token_id=1,
                                         eos_token_id=2, pad_token_id=0)).cpu().eval()
        # These would invalidate raw scoring if inherited into the fresh config.
        model.generation_config.repetition_penalty = 1.9
        model.generation_config.min_length = 20
        model.generation_config.forced_eos_token_id = 2
        model.generation_config.suppress_tokens = [5]
        model.generation_config.sequence_bias = {(6,): -5.0}
        model.generation_config.renormalize_logits = True
        config = m._neutral_generation_config(GenerationConfig, model.generation_config.to_dict(), 6)
        recorder = m._make_recorder(torch, LogitsProcessor)
        input_ids = torch.tensor([[1, 3, 4]])
        actual_processors = []
        original = model._get_logits_processor

        def inspect_processors(*args, **kwargs):
            processors = original(*args, **kwargs)
            actual_processors.extend(processors)
            return processors

        # Config-only GPT-2: no from_pretrained, tokenizer, hub, or GPU access.
        with torch.inference_mode(), patch.object(model, "_get_logits_processor", side_effect=inspect_processors):
            output = model.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                                    generation_config=config, use_model_defaults=False,
                                    logits_processor=[recorder], synced_gpus=False,
                                    return_dict_in_generate=True, output_scores=True, output_logits=True)
        actual = output.sequences[0, input_ids.shape[1]:].tolist()
        recorded = torch.tensor(recorder.finish(actual))
        expected = torch.stack([scores.log_softmax(-1)[0, token]
                                for scores, token in zip(output.scores, actual)])
        self.assertEqual(actual_processors, [recorder])
        self.assertGreater(len(actual), 0)
        torch.testing.assert_close(recorded, expected, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(recorded.exp(), expected.exp(), atol=1e-7, rtol=1e-6)
        for processed, raw in zip(output.scores, output.logits):
            torch.testing.assert_close(processed, raw, atol=0, rtol=0)
        self.assertTrue(all(x.numel() == 1 and not x.requires_grad
                            for x in recorder.ids + recorder.logprobs + recorder.finite))
        wrong = actual.copy()
        wrong[0] = (wrong[0] + 1) % 32
        with self.assertRaisesRegex(RuntimeError, "argmax"):
            recorder.finish(wrong)
        with self.assertRaisesRegex(RuntimeError, "length"):
            recorder.finish(actual[:-1])


if __name__ == "__main__":
    unittest.main()
