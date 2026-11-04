"""
LemGendary Dataset Compiler: generator strategies and degradation configs.

The compile CLI offers exactly the strategies the generators accept, and a
degradation chain reports the configured parameters of every kernel (also
written to a synthetic manifold's ``dataset_info.yaml``).
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from core.cli_args import build_parser
from degrade import GaussianBlur, LowLight, parse_profile
from generators import Generator, LabelGenerator, MaskGenerator, PromptGenerator
from tools.generate_degrade_registry import ManifoldDescription, write_metadata


def _accepts(parser: argparse.ArgumentParser, flag: str, value: str) -> bool:
    """True when ``parser`` accepts ``flag value`` (argparse exits on a bad choice)."""
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            parser.parse_args([flag, value])
    except SystemExit:
        return False
    return True


class TestGeneratorStrategies(unittest.TestCase):
    """``strategies()`` is the single list of accepted names."""

    def test_cli_choices_come_from_the_generators(self) -> None:
        """--label/--mask/--prompt-strategy accept exactly the generator strategies."""
        parser = build_parser()
        for flag, generator in (("--label-strategy", LabelGenerator),
                                ("--mask-strategy", MaskGenerator),
                                ("--prompt-strategy", PromptGenerator)):
            for name in generator.strategies():
                self.assertTrue(_accepts(parser, flag, name), (flag, name))
            self.assertFalse(_accepts(parser, flag, "not-a-strategy"), flag)

    def test_every_listed_strategy_constructs(self) -> None:
        """Each listed name builds a generator that satisfies the protocol."""
        for name in LabelGenerator.strategies():
            self.assertIsInstance(LabelGenerator(name), Generator)
        for name in MaskGenerator.strategies():
            self.assertIsInstance(MaskGenerator(name), Generator)
        for name in PromptGenerator.strategies():
            self.assertIsInstance(PromptGenerator(name), Generator)


class TestDegradationConfig(unittest.TestCase):
    """Kernels are dataclasses whose clamped fields form ``config()``."""

    def test_kernel_config_is_clamped_constructor_arguments(self) -> None:
        """config() reports the constructor arguments after clamping."""
        self.assertEqual(GaussianBlur(sigma=0.0, kernel_size=4).config(),
                         {"sigma": 0.1, "kernel_size": 5})
        self.assertEqual(LowLight(gamma=0.5).config()["gamma"], 1.0)

    def test_dataset_info_records_the_chain(self) -> None:
        """write_metadata stores each step's type and parameters."""
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            desc = ManifoldDescription(out, out / "div2k", "vintage-film", "high", 7)
            write_metadata(desc, [])
            info = json.loads((out / "dataset_info.yaml").read_text(encoding="utf-8"))
        expected = parse_profile("vintage-film", intensity="high").config()["steps"]
        self.assertEqual(info["degradation_steps"], expected)
        self.assertEqual([s["type"] for s in expected],
                         ["film_grain", "film_scratches", "color_fade"])


if __name__ == "__main__":
    unittest.main()
