import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OWNED = re.compile(r"^\s*(def (compute_vtrace|compute_caps_loss\w*|impala_loss|impala_targets|sil_(?:confident_gauss|positive_advantage|mc_returns|policy_kl|separate_step|clipped_step|full_cutoff|row_normalize|ensure|stats)|compute_sil_loss\w*|gm_cap_probs)\b|class (RSSM|GoldenMemory|PopArtLayer|PopArtAdapter|RunningMeanStd|EntropyController|FixedEntropy)\b)")


class PackageOwnedNames(unittest.TestCase):
    def test_package_names_are_not_defined_in_the_project(self):
        hits = []
        for p in (ROOT / "main").rglob("*.py"):
            for i, line in enumerate(p.read_text().split("\n"), 1):
                if OWNED.match(line):
                    hits.append(f"{p.relative_to(ROOT)}:{i}: {line.strip()}")
        self.assertEqual(hits, [])


if __name__ == "__main__":
    unittest.main()
