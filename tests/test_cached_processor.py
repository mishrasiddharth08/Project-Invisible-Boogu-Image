import json
from pathlib import Path
from unittest.mock import patch
from huggingface_hub.errors import LocalEntryNotFoundError
from pi_boogu_image.lib import assets, pipeline
import pytest


def test_complete_pinned_offline_processor_beats_incomplete_bundle(tmp_path):
    repo = "Boogu/Boogu-Image-0.1-Base"
    pins = json.loads((assets.EXT / "sources.json").read_text(encoding="utf-8"))
    selected = tmp_path / "snapshot" / "processor"
    selected.mkdir(parents=True)
    for name in pins["processor_files"]:
        (selected / Path(name).name).write_text("fixture")
    with patch.object(pipeline, "CONFIGS", tmp_path / "incomplete"), patch.object(assets, "roots", return_value=[]), \
         patch("huggingface_hub.snapshot_download", return_value=str(selected.parent)) as cached:
        assert pipeline.processor_path() == selected
        cached.assert_called_once_with(repo_id=repo, revision=pins["models"][repo], local_files_only=True,
                                       allow_patterns=["processor/*"])
        cached.reset_mock()
        cached.side_effect = [LocalEntryNotFoundError("Forge cache is empty"), str(selected.parent)]
        assert pipeline.processor_path() == selected
        assert cached.call_args.kwargs["cache_dir"] == Path.home() / ".cache" / "huggingface" / "hub"
        cached.side_effect = None
        (selected / "tokenizer.json").unlink()
        with pytest.raises(RuntimeError, match="processor is missing"):
            pipeline.processor_path()
        cached.side_effect = LocalEntryNotFoundError("not cached")
        with pytest.raises(RuntimeError, match="processor is missing"):
            pipeline.processor_path()
