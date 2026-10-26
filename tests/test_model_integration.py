"""Check all real timm architectures without downloading or allocating weights.

Meta tensors validate architecture construction and tensor shapes, not numerical
inference or accuracy. The pretrained regression check also exercises timm's real
Hub imports and source resolution, stopping at the download boundary. timm 1.0.11
reads its small stochastic-depth schedule with Tensor.item(), which meta tensors
do not support. Only that schedule is created
on CPU by the test helper; every model parameter stays on the meta device.
"""

import importlib.util
import unittest
from unittest.mock import patch

from fan_classifier.registry import build_model, list_model_specs


class WeightDownloadReached(Exception):
    """Stop before downloading public weights, after resolving their source."""


@unittest.skipUnless(
    all(importlib.util.find_spec(name) for name in ("torch", "timm")),
    "training dependencies are not installed",
)
class RealModelShapeTests(unittest.TestCase):
    def test_all_six_default_pretrained_models_reach_their_hub_repository(self):
        import torch

        original_linspace = torch.linspace

        def schedule_on_cpu(*args, **kwargs):
            if len(args) == 3 and args[0] == 0 and not kwargs:
                self.assertLessEqual(args[2], 128)
                kwargs["device"] = "cpu"
            return original_linspace(*args, **kwargs)

        for spec in list_model_specs():
            with self.subTest(model=spec.alias):
                # Keep has_hf_hub, source resolution and load_pretrained real.
                # Mocking any of those would hide incompatible Hub imports.
                with (
                    patch("torch.linspace", side_effect=schedule_on_cpu),
                    patch(
                        "timm.models._hub.hf_hub_download",
                        side_effect=WeightDownloadReached,
                    ) as download,
                    torch.device("meta"),
                    torch.inference_mode(),
                    self.assertRaises(WeightDownloadReached),
                ):
                    build_model(spec.alias)

                download.assert_called_once()
                args, kwargs = download.call_args
                repository = kwargs.get("repo_id", args[0] if args else None)
                self.assertEqual(repository, f"timm/{spec.timm_name}")
                self.assertIn(kwargs["filename"], ("model.safetensors", "pytorch_model.bin"))
                self.assertIsNone(kwargs["revision"])

    def test_all_six_architectures_accept_their_configured_resolution(self):
        import torch

        original_linspace = torch.linspace

        def schedule_on_cpu(*args, **kwargs):
            # timm's depth-sized schedule is the only data-dependent constructor
            # operation. Keep all other tensors subject to the meta context.
            if len(args) == 3 and args[0] == 0 and not kwargs:
                self.assertLessEqual(args[2], 128)
                kwargs["device"] = "cpu"
            return original_linspace(*args, **kwargs)

        with (
            patch("torch.linspace", side_effect=schedule_on_cpu),
            patch(
                "timm.models._builder.load_pretrained",
                side_effect=AssertionError("No weight downloads in tests"),
            ),
        ):
            for spec in list_model_specs():
                with self.subTest(model=spec.alias), torch.device("meta"), torch.inference_mode():
                    model = build_model(spec.alias, pretrained=False)
                    self.assertTrue(
                        all(parameter.device.type == "meta" for parameter in model.parameters())
                    )
                    model.eval()
                    synthetic_input = torch.empty(1, 3, spec.image_size, spec.image_size)
                    logits = model(synthetic_input)
                    self.assertEqual(tuple(logits.shape), (1, 2))
                    self.assertEqual(logits.device.type, "meta")
                    del model, synthetic_input, logits


if __name__ == "__main__":
    unittest.main()
