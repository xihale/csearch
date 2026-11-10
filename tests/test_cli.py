import argparse
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

from csearch.cli import _use_local, _positional_args, _report, _first_line, _print_hit_source, main
from csearch.completions import BASH, ZSH, FISH


class TestCLIQueryDirectoryResolution(unittest.TestCase):
    """Test CLI positional argument disambiguation when query matches directory name."""

    def test_bare_query_matching_directory(self):
        # 'models' is an existing directory in the repo
        self.assertTrue(Path("models").is_dir())
        with patch("csearch.cli.cmd_search", return_value=0) as mock_search:
            rc = main(["models"])
            self.assertEqual(rc, 0)
            mock_search.assert_called_once()
            ns = mock_search.call_args[0][0]
            self.assertEqual(ns.root, ".")
            self.assertEqual(ns.query, "models")

    def test_bare_query_matching_directory_with_flags(self):
        with patch("csearch.cli.cmd_search", return_value=0) as mock_search:
            rc = main(["models", "-k", "3"])
            self.assertEqual(rc, 0)
            mock_search.assert_called_once()
            ns = mock_search.call_args[0][0]
            self.assertEqual(ns.root, ".")
            self.assertEqual(ns.query, "models")
            self.assertEqual(ns.k, 3)

    def test_search_alias_s_with_directory_query(self):
        with patch("csearch.cli.cmd_search", return_value=0) as mock_search:
            rc = main(["s", "models"])
            self.assertEqual(rc, 0)
            mock_search.assert_called_once()
            ns = mock_search.call_args[0][0]
            self.assertEqual(ns.root, ".")
            self.assertEqual(ns.query, "models")

    def test_search_subcommand_with_directory_query(self):
        with patch("csearch.cli.cmd_search", return_value=0) as mock_search:
            rc = main(["search", "models"])
            self.assertEqual(rc, 0)
            mock_search.assert_called_once()
            ns = mock_search.call_args[0][0]
            self.assertEqual(ns.root, ".")
            self.assertEqual(ns.query, "models")

    def test_explicit_root_and_directory_query(self):
        with patch("csearch.cli.cmd_search", return_value=0) as mock_search:
            rc = main([".", "models"])
            self.assertEqual(rc, 0)
            mock_search.assert_called_once()
            ns = mock_search.call_args[0][0]
            self.assertEqual(ns.root, ".")
            self.assertEqual(ns.query, "models")

    def test_explicit_directory_root_and_query(self):
        with patch("csearch.cli.cmd_search", return_value=0) as mock_search:
            rc = main(["models", "some_code"])
            self.assertEqual(rc, 0)
            mock_search.assert_called_once()
            ns = mock_search.call_args[0][0]
            self.assertEqual(ns.root, "models")
            self.assertEqual(ns.query, "some_code")

    def test_query_matching_tests_directory(self):
        self.assertTrue(Path("tests").is_dir())
        with patch("csearch.cli.cmd_search", return_value=0) as mock_search:
            rc = main(["tests", "-k", "2"])
            self.assertEqual(rc, 0)
            mock_search.assert_called_once()
            ns = mock_search.call_args[0][0]
            self.assertEqual(ns.root, ".")
            self.assertEqual(ns.query, "tests")
            self.assertEqual(ns.k, 2)


class TestPositionalArgsHelper(unittest.TestCase):
    def test_extract_positional(self):
        self.assertEqual(_positional_args(["models"]), ["models"])
        self.assertEqual(_positional_args(["models", "-k", "3"]), ["models"])
        self.assertEqual(_positional_args(["-k", "3", "models"]), ["models"])
        self.assertEqual(_positional_args(["--model", "/tmp/m", "models"]), ["models"])
        self.assertEqual(_positional_args(["--model=/tmp/m", "models"]), ["models"])
        self.assertEqual(_positional_args(["--backend", "torch", "models"]), ["models"])
        self.assertEqual(_positional_args(["--variant", "int8", "models"]), ["models"])
        self.assertEqual(_positional_args(["--dim", "512", "models"]), ["models"])
        self.assertEqual(_positional_args(["--batch", "32", "models"]), ["models"])
        self.assertEqual(_positional_args(["--cpu", "models"]), ["models"])
        self.assertEqual(_positional_args(["--local", "models"]), ["models"])
        self.assertEqual(_positional_args(["--multimodal", "models"]), ["models"])
        self.assertEqual(_positional_args(["--show", "models"]), ["models"])
        self.assertEqual(_positional_args(["--", "models", "-k"]), ["models", "-k"])


class TestUseLocal(unittest.TestCase):
    def setUp(self):
        self.defaults = dict(
            local=False,
            model=None,
            backend="ort",
            variant=None,
            batch=16,
            cpu=False,
        )

    def test_defaults_return_false(self):
        ns = argparse.Namespace(**self.defaults)
        self.assertFalse(_use_local(ns))

    def test_empty_namespace_returns_false(self):
        self.assertFalse(_use_local(argparse.Namespace()))

    def test_cpu_flag(self):
        ns = argparse.Namespace(**{**self.defaults, "cpu": True})
        self.assertTrue(_use_local(ns))

    def test_variant_flag(self):
        ns = argparse.Namespace(**{**self.defaults, "variant": "int8"})
        self.assertTrue(_use_local(ns))

    def test_backend_flag(self):
        ns = argparse.Namespace(**{**self.defaults, "backend": "torch"})
        self.assertTrue(_use_local(ns))

    def test_batch_flag(self):
        ns = argparse.Namespace(**{**self.defaults, "batch": 8})
        self.assertTrue(_use_local(ns))

    def test_model_flag(self):
        ns = argparse.Namespace(**{**self.defaults, "model": "/tmp/m"})
        self.assertTrue(_use_local(ns))

    def test_local_flag(self):
        ns = argparse.Namespace(**{**self.defaults, "local": True})
        self.assertTrue(_use_local(ns))


class TestShellCompletions(unittest.TestCase):
    def test_all_completions_contain_local(self):
        for shell_name, script in [("BASH", BASH), ("ZSH", ZSH), ("FISH", FISH)]:
            with self.subTest(shell=shell_name):
                self.assertIn("--local", script)

    def test_all_completions_contain_multimodal(self):
        for shell_name, script in [("BASH", BASH), ("ZSH", ZSH), ("FISH", FISH)]:
            with self.subTest(shell=shell_name):
                self.assertIn("--multimodal", script)

    def test_bash_subcommand_aliases(self):
        self.assertIn("index|update|search|status|stats|completions|i|u|s|st", BASH)
        self.assertIn("i)  sub=index ;;", BASH)
        self.assertIn("u)  sub=update ;;", BASH)
        self.assertIn("s)  sub=search ;;", BASH)
        self.assertIn("st) sub=status ;;", BASH)

    def test_zsh_subcommand_aliases(self):
        self.assertIn("index|update|i|u)", ZSH)
        self.assertIn("search|s)", ZSH)
        self.assertIn("status|stats|st)", ZSH)

    def test_fish_subcommand_aliases(self):
        self.assertIn("__fish_seen_subcommand_from index update search i u s", FISH)
        self.assertIn("-l local -d 'embed in this process instead of the shared daemon'", FISH)
        self.assertIn("-l multimodal -d 'experimental: also index images and PDFs'", FISH)


class TestReportAndFirstLine(unittest.TestCase):
    def test_report_uses_gitignore_key_when_present(self):
        stats = {
            "files": 10,
            "chunks": 25,
            "reembedded_chunks": 5,
            "changed_files": 2,
            "reused_chunks": 20,
            "encode_seconds": 1.2,
            "gitignore": "custom gitignore message",
        }
        with patch("csearch.cli._ensure_gitignored") as mock_ensure:
            with patch("builtins.print") as mock_print:
                _report(Path("."), stats)
                mock_ensure.assert_not_called()
                mock_print.assert_any_call("  gitignore   custom gitignore message")

    def test_report_calls_ensure_gitignored_when_missing(self):
        stats = {
            "files": 10,
            "chunks": 25,
            "reembedded_chunks": 5,
            "changed_files": 2,
            "reused_chunks": 20,
            "encode_seconds": 1.2,
        }
        with patch("csearch.cli._ensure_gitignored", return_value="added .csearch/") as mock_ensure:
            with patch("builtins.print") as mock_print:
                _report(Path("."), stats)
                mock_ensure.assert_called_once_with(Path("."))
                mock_print.assert_any_call("  gitignore   added .csearch/")


class TestPrintHitSource(unittest.TestCase):
    def test_print_hit_source_image(self):
        root = Path("/fake/root")
        h = {"path": "docs/arch.png", "kind": "image", "preview": "[image 64x64]", "start": 0, "end": 1}
        with patch("builtins.print") as mock_print:
            _print_hit_source(root, h)
            mock_print.assert_called_once_with("      [image 64x64]")

    def test_print_hit_source_image_by_extension(self):
        root = Path("/fake/root")
        h = {"path": "docs/arch.webp", "start": 0, "end": 1}
        with patch("csearch.cli._first_line", return_value="[image 100x100]"):
            with patch("builtins.print") as mock_print:
                _print_hit_source(root, h)
                mock_print.assert_called_once_with("      [image 100x100]")

    def test_print_hit_source_pdf(self):
        root = Path("/fake/root")
        h = {"path": "docs/spec.pdf", "kind": "pdf", "preview": "[pdf p.1] Title", "start": 0, "end": 1}
        with patch("csearch.cli.indexer._extract_pdf_pages", return_value=[(1, "Title\nSection A\n")]):
            with patch("builtins.print") as mock_print:
                _print_hit_source(root, h)
                self.assertEqual(mock_print.call_count, 2)
                mock_print.assert_any_call("      p.1  Title")
                mock_print.assert_any_call("      p.1  Section A")

    def test_print_hit_source_pdf_fallback(self):
        root = Path("/fake/root")
        h = {"path": "docs/spec.pdf", "kind": "pdf", "preview": "[pdf p.2] Title", "start": 1, "end": 2}
        with patch("csearch.cli.indexer._extract_pdf_pages", return_value=[]):
            with patch("builtins.print") as mock_print:
                _print_hit_source(root, h)
                mock_print.assert_called_once_with("      [pdf p.2] Title")


if __name__ == "__main__":
    unittest.main()
