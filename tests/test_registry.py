import sys
import types
import unittest
from unittest.mock import Mock, patch

from fan_classifier.registry import build_model, get_model_spec, list_model_specs


class RegistryTests(unittest.TestCase):
    def test_aliases_and_full_identifiers_resolve_to_same_spec(self):
        self.assertEqual(len(list_model_specs()), 6)
        for spec in list_model_specs():
            self.assertIs(get_model_spec(spec.alias), get_model_spec(spec.timm_name))

    def test_factory_routes_every_model_to_correct_backbone_and_two_class_head(self):
        factory = Mock()
        with patch.dict(sys.modules, {"timm": types.SimpleNamespace(create_model=factory)}):
            for spec in list_model_specs():
                model = build_model(spec.alias, pretrained=False)
                factory.assert_called_with(spec.timm_name, pretrained=False, num_classes=2)
                self.assertIs(model, factory.return_value)

    def test_unknown_name_never_falls_back_to_another_model(self):
        with self.assertRaisesRegex(ValueError, "Unknown model"):
            build_model("efficientnet")

    def test_invalid_head_size_is_rejected(self):
        for value in (0, -1, True, 2.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                build_model("eva_large_196", num_classes=value)


if __name__ == "__main__":
    unittest.main()
