import ast
import fnmatch
import importlib.util
import re
import unittest
from unittest import mock
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def load_vision_tower_utils():
    module_path = REPOSITORY_ROOT / "llava" / "model" / "multimodal_encoder" / "vision_tower_utils.py"
    spec = importlib.util.spec_from_file_location("llava_vision_tower_utils_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class VisionTowerRoutingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_vision_tower_utils()

    def test_existing_siglip_path_is_not_routed_to_clip(self):
        directory = Path("/checkpoints/siglip-so400m-patch14-384")
        with mock.patch.object(self.module.os.path, "exists", side_effect=AssertionError("generic path check ran")):
            tower_path, tower_type = self.module.resolve_vision_tower(directory)

        self.assertEqual(tower_path, str(directory))
        self.assertEqual(tower_type, "siglip")

    def test_existing_generic_path_uses_clip_fallback(self):
        directory = Path("/checkpoints/clip-vit-large-patch14-336")
        with mock.patch.object(self.module.os.path, "exists", return_value=True):
            _, tower_type = self.module.resolve_vision_tower(directory)

        self.assertEqual(tower_type, "clip")

    def test_none_has_an_actionable_error(self):
        with self.assertRaisesRegex(ValueError, "vision tower"):
            self.module.resolve_vision_tower(None)


class QFormerCompatibilitySourceTest(unittest.TestCase):
    def test_new_and_legacy_transformers_helper_locations_are_supported(self):
        qformer_path = REPOSITORY_ROOT / "llava" / "model" / "multimodal_resampler" / "qformer.py"
        tree = ast.parse(qformer_path.read_text(encoding="utf-8"), filename=str(qformer_path))
        helper_names = {
            "apply_chunking_to_forward",
            "find_pruneable_heads_and_indices",
            "prune_linear_layer",
        }
        compatibility_try = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Try)
            and any(
                isinstance(child, ast.ImportFrom) and child.module == "transformers.pytorch_utils"
                for statement in node.body
                for child in ast.walk(statement)
            )
        )
        current_imports = {
            alias.name
            for statement in compatibility_try.body
            for node in ast.walk(statement)
            if isinstance(node, ast.ImportFrom) and node.module == "transformers.pytorch_utils"
            for alias in node.names
        }
        legacy_imports = {
            alias.name
            for handler in compatibility_try.handlers
            for node in ast.walk(handler)
            if isinstance(node, ast.ImportFrom) and node.module == "transformers.modeling_utils"
            for alias in node.names
        }

        self.assertTrue(helper_names <= current_imports)
        self.assertTrue(helper_names <= legacy_imports)
        self.assertTrue(
            any(isinstance(handler.type, ast.Name) and handler.type.id == "ImportError" for handler in compatibility_try.handlers)
        )


class CurrentEnvironmentPackagingTest(unittest.TestCase):
    def test_package_discovery_does_not_expose_vendored_trl(self):
        pyproject = (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        section = re.search(
            r"\[tool\.setuptools\.packages\.find\](.*?)(?=\n\[|\Z)",
            pyproject,
            flags=re.DOTALL,
        ).group(1)
        include_value = re.search(r"include\s*=\s*\[(.*?)\]", section, flags=re.DOTALL).group(1)
        include_patterns = re.findall(r'["\']([^"\']+)["\']', include_value)

        self.assertTrue(any(fnmatch.fnmatchcase("llava", pattern) for pattern in include_patterns))
        self.assertTrue(any(fnmatch.fnmatchcase("llava.model.language_model", pattern) for pattern in include_patterns))
        self.assertFalse(any(fnmatch.fnmatchcase("llava-critic-r1", pattern) for pattern in include_patterns))
        self.assertFalse(any(fnmatch.fnmatchcase("trl", pattern) for pattern in include_patterns))


class ConversationTokenizerSourceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        conversation_path = REPOSITORY_ROOT / "llava" / "conversation.py"
        cls.tree = ast.parse(conversation_path.read_text(encoding="utf-8"), filename=str(conversation_path))

    @classmethod
    def conversation_call(cls, variable_name):
        assignment = next(
            node
            for node in cls.tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == variable_name for target in node.targets)
        )
        return assignment.value

    def test_conversation_import_has_no_eager_tokenizer_loading(self):
        transformer_imports = [
            node
            for node in ast.walk(self.tree)
            if isinstance(node, ast.ImportFrom) and node.module == "transformers"
        ]
        from_pretrained_calls = [
            node
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "from_pretrained"
        ]

        self.assertFalse(transformer_imports)
        self.assertFalse(from_pretrained_calls)

    def test_llama3_template_is_kept_with_an_explicit_none_tokenizer(self):
        llama3_call = self.conversation_call("conv_llava_llama_3")
        keywords = {keyword.arg: keyword.value for keyword in llama3_call.keywords}

        self.assertEqual(ast.literal_eval(keywords["tokenizer_id"]), "meta-llama/Meta-Llama-3-8B-Instruct")
        self.assertIsNone(ast.literal_eval(keywords["tokenizer"]))

    def test_llama3_missing_tokenizer_error_explains_how_to_inject_it(self):
        get_prompt = next(
            node
            for node in ast.walk(self.tree)
            if isinstance(node, ast.FunctionDef) and node.name == "get_prompt"
        )
        missing_tokenizer_guard = next(
            node
            for node in ast.walk(get_prompt)
            if isinstance(node, ast.If) and ast.unparse(node.test) == "self.tokenizer is None"
        )
        error = next(node for node in ast.walk(missing_tokenizer_guard) if isinstance(node, ast.Raise))
        message = ast.literal_eval(error.exc.args[0])

        self.assertIn("conversation.tokenizer", message)
        self.assertIn("get_prompt", message)
        self.assertNotIn("permissions", message.lower())

    def test_qwen_1_5_template_does_not_depend_on_a_tokenizer(self):
        qwen_call = self.conversation_call("conv_qwen")
        self.assertNotIn("tokenizer", {keyword.arg for keyword in qwen_call.keywords})

        registry_assignment = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "conv_templates" for target in node.targets)
        )
        registry = {
            ast.literal_eval(key): value.id
            for key, value in zip(registry_assignment.value.keys, registry_assignment.value.values)
            if isinstance(key, ast.Constant) and isinstance(value, ast.Name)
        }
        self.assertEqual(registry["qwen_1_5"], "conv_qwen")


class SigLipListForwardSourceTest(unittest.TestCase):
    def test_list_branch_checks_each_feature_tensor(self):
        encoder_path = REPOSITORY_ROOT / "llava" / "model" / "multimodal_encoder" / "siglip_encoder.py"
        tree = ast.parse(encoder_path.read_text(encoding="utf-8"), filename=str(encoder_path))
        image_loop = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == "image"
        )
        assertions = [ast.unparse(node.test) for node in ast.walk(image_loop) if isinstance(node, ast.Assert)]

        self.assertIn("image_feature.shape[-2] == 729", assertions)
        self.assertNotIn("image_features.shape[-2] == 729", assertions)


class MultimodalBatchSourceTest(unittest.TestCase):
    def test_single_default_modality_is_expanded_to_the_batch(self):
        architecture_path = REPOSITORY_ROOT / "llava" / "model" / "llava_arch.py"
        source = architecture_path.read_text(encoding="utf-8")

        self.assertIn("modalities = modalities * batch_size", source)
        self.assertIn("Expected one modality per batch item", source)


if __name__ == "__main__":
    unittest.main()
