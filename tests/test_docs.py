"""Guard against documentation drifting away from the program: every `--flag` used in a command shown in
docs/QUICKSTART.md or README.md must exist in the real command line, and every subcommand shown must be real."""
import re
import unittest
from pathlib import Path

from hust_eval_mmlu_pro import cli

ROOT = Path(__file__).resolve().parents[1]
DOCS = [ROOT / "README.md", ROOT / "docs" / "QUICKSTART.md"]


def _walk(parser, prefix=()):
    """Yield (subcommand path tuple, set of option strings) for every parser, depth first."""
    import argparse
    flags = {o for a in parser._actions for o in a.option_strings}
    yield prefix, flags
    for a in parser._actions:
        if isinstance(a, argparse._SubParsersAction):
            for name, sub in a.choices.items():
                yield from _walk(sub, prefix + (name,))


def _commands(text):
    """Join backslash-continued lines of fenced code blocks and return the lines that start our command."""
    out = []
    for block in re.findall(r"```(?:bash)?\n(.*?)```", text, re.S):
        joined = re.sub(r"\\\n\s*", " ", block)
        for line in joined.splitlines():
            line = line.strip()
            line = re.sub(r"^nohup\s+", "", line)
            if line.startswith("hust-eval-mmlu-pro "):
                out.append(line)
    return out


class DocsMatchProgram(unittest.TestCase):
    def test_every_documented_command_and_flag_exists(self):
        table = dict(_walk(cli.build_parser()))
        seen = 0
        for doc in DOCS:
            for line in _commands(doc.read_text()):
                tokens = re.split(r"\s+", line)[1:]
                # longest known subcommand path made of the leading non-flag tokens
                path = ()
                for t in tokens:
                    if t.startswith("-") or t.startswith("<") or t.startswith("$"):
                        break
                    if path + (t,) in table:
                        path += (t,)
                    else:
                        break
                if tokens and tokens[0] in ("--version", "--help"):
                    continue
                self.assertIn(path, table, "%s: unknown command in: %s" % (doc.name, line))
                self.assertTrue(path, "%s: no subcommand in: %s" % (doc.name, line))
                for t in tokens:
                    if t.startswith("--"):
                        flag = t.split("=")[0]
                        self.assertIn(flag, table[path], "%s: %s has no flag %s" % (doc.name, " ".join(path), flag))
                seen += 1
        self.assertGreater(seen, 10, "expected to check a meaningful number of documented commands")


if __name__ == "__main__":
    unittest.main()
