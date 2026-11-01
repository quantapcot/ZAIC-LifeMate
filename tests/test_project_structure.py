"""Guard the project boundaries documented in README.md."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ProjectStructureTest(unittest.TestCase):
    def test_required_directories_exist(self) -> None:
        required = (
            "src/asr",
            "src/agent",
            "src/training",
            "src/evaluation",
            "configs",
            "scripts",
            "train",
            "public_test",
            "models/pretrained",
            "models/checkpoints",
            "outputs/benchmarks",
            "tests",
            "docs",
        )

        for relative_path in required:
            with self.subTest(path=relative_path):
                self.assertTrue((ROOT / relative_path).is_dir())

    def test_pipeline_entry_points_are_in_scripts(self) -> None:
        entry_points = (
            "run_agent.py",
            "run_plan.py",
            "transcribe_audio.py",
            "finalize_submission.py",
        )

        for filename in entry_points:
            with self.subTest(filename=filename):
                self.assertTrue((ROOT / "scripts" / filename).is_file())


if __name__ == "__main__":
    unittest.main()
