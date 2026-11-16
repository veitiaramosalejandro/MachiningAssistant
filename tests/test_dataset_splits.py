"""Checks for user-edited split tables shared by both model workflows."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Dataset.Preprocessing.dataset_splits import read_split_csv, split_recording_ids, write_split_csv


def rows():
    return [
        {"recording_id": "record_a", "source_recording_id": "source_a", "split": "train"},
        {"recording_id": "record_b", "source_recording_id": "source_b", "split": "val"},
        {"recording_id": "record_c", "source_recording_id": "source_c", "split": "test"},
    ]


def test_second_read_uses_edited_csv_without_reimport(tmp_path):
    path = tmp_path / "splits.csv"
    data = rows()
    write_split_csv(path, data)
    first = split_recording_ids(read_split_csv(path))
    data[0]["split"], data[2]["split"] = "test", "train"
    write_split_csv(path, data)
    second = split_recording_ids(read_split_csv(path))
    assert first["train"] == ["record_a"]
    assert second["train"] == ["record_c"]
    assert second["test"] == ["record_a"]


def test_unicode_csv_and_excluded_clip_preserve_source_group(tmp_path):
    path = tmp_path / "splits.csv"
    data = rows()
    data[0]["recording_id"] = "01 - Abertura de caixa - início"
    data.append({"recording_id": "Medição de ferramenta", "source_recording_id": "source_a", "split": "exclude"})
    write_split_csv(path, data)
    loaded = read_split_csv(path)
    assert loaded == data
    assert split_recording_ids(loaded)["train"] == [data[0]["recording_id"]]


def test_clips_from_same_source_cannot_cross_splits(tmp_path):
    path = tmp_path / "splits.csv"
    data = rows()
    data.append({"recording_id": "another_clip", "source_recording_id": "source_a", "split": "test"})
    write_split_csv(path, data)
    with pytest.raises(ValueError, match="source_a.*both train and test"):
        read_split_csv(path)
    data[-1]["split"] = "train"
    write_split_csv(path, data)
    assert split_recording_ids(read_split_csv(path))["train"] == ["record_a", "another_clip"]


def test_duplicate_recording_is_not_counted_twice(tmp_path):
    path = tmp_path / "splits.csv"
    data = rows()
    data.append(dict(data[0]))
    write_split_csv(path, data)
    with pytest.raises(ValueError, match="duplicate recording_id"):
        read_split_csv(path)


@pytest.mark.parametrize("split", ["validation", "", "trian"])
def test_invalid_or_blank_assignment_is_reported(tmp_path, split):
    path = tmp_path / "splits.csv"
    data = rows()
    data[1]["split"] = split
    write_split_csv(path, data)
    with pytest.raises(ValueError, match="split"):
        read_split_csv(path)


def test_excluding_an_entire_split_is_reported(tmp_path):
    path = tmp_path / "splits.csv"
    data = rows()
    data[2]["split"] = "exclude"
    write_split_csv(path, data)
    with pytest.raises(ValueError, match="empty data splits: test"):
        read_split_csv(path)


def test_wrong_table_headers_are_reported(tmp_path):
    path = tmp_path / "splits.csv"
    path.write_text("recording_id,split\nrecord_a,train\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing CSV columns: source_recording_id"):
        read_split_csv(path)
