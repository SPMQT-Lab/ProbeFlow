from pathlib import Path

import pytest

from probeflow.gui.browse.file_actions import create_folder, move_files


def test_create_folder_rejects_path_components(tmp_path):
    with pytest.raises(ValueError):
        create_folder(tmp_path, "nested/folder")

    created = create_folder(tmp_path, "sorted")
    assert created == tmp_path / "sorted"
    assert created.is_dir()


def test_move_files_moves_sources_and_preserves_names(tmp_path):
    destination = tmp_path / "sorted"
    destination.mkdir()
    sources = [tmp_path / "a.sxm", tmp_path / "b.sm4"]
    for source in sources:
        source.write_bytes(b"data")

    moved = move_files(sources, destination)

    assert moved == [destination / source.name for source in sources]
    assert all(path.exists() for path in moved)
    assert not any(path.exists() for path in sources)


def test_move_files_rejects_collisions_and_same_folder(tmp_path):
    source = tmp_path / "scan.sxm"
    source.write_bytes(b"data")
    destination = tmp_path / "sorted"
    destination.mkdir()
    (destination / source.name).write_bytes(b"existing")

    with pytest.raises(FileExistsError):
        move_files([source], destination)
    with pytest.raises(ValueError):
        move_files([source], tmp_path)
    assert source.read_bytes() == b"data"
