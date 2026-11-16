"""Create lossless one-second WAV clips and labels for the opening-pocket demo."""

import argparse
import csv
import json
import os
import wave
from pathlib import Path


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path,
                        default=Path(__file__).resolve().parent.parent/'Data'/'01 - Abertura de caixa - início')
    parser.add_argument('--output', type=Path, default=None,
                        help='New output directory; default: window_demo under the source sample')
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve() if args.output is not None else source/'window_demo'
    audio_path = source / "audio_000000.wav"
    events = read_json(source / "events.json")
    playback = read_json(source / "playback.json")
    metadata = read_json(source / "metadata.json")
    annotations = events["annotations"]

    # This demonstration deliberately targets the supplied single-event clip.
    # A different event timeline or playback offset needs a new alignment step.
    assert events["time_reference"] == "playback_seconds"
    assert playback["AudioOffsetMs"] == 0, "Playback-to-WAV time alignment must be resolved first."
    assert len(annotations) == 1 and annotations[0]["timestamp_seconds"] == 0
    event = annotations[0]
    assert event["event_changes"]["machine_state"] is True
    assert event["event_changes"]["quality"] is True
    assert event["machine_state"] == "machining" and event["quality"] == 3

    with wave.open(str(audio_path), "rb") as original:
        rate = original.getframerate()
        channels = original.getnchannels()
        width = original.getsampwidth()
        frames = original.getnframes()
        compression = original.getcomptype()
        pcm = original.readframes(frames)
    assert (rate, channels, width, compression) == (44100, 2, 2, "NONE")
    assert frames == 2730672
    frame_bytes = channels * width
    assert len(pcm) == frames * frame_bytes
    original_source = metadata["SourceRecording"]["Folder"].replace("\\", "/").rstrip("/").split("/")[-1]

    clips = output / "clips"
    if clips.exists():
        raise FileExistsError("The clips folder already exists; inspect it before regenerating.")
    output.mkdir(parents=True, exist_ok=True)
    clips.mkdir()
    rows = []
    for index, start in enumerate(range(0, frames, rate)):
        end = min(start + rate, frames)
        name = f"clip_{index:04d}.wav"
        payload = pcm[start * frame_bytes:end * frame_bytes]
        with wave.open(str(clips / name), "wb") as clip:
            clip.setnchannels(channels)
            clip.setsampwidth(width)
            clip.setframerate(rate)
            clip.writeframes(payload)
        rows.append({
            "clip_id": index,
            "wav_file": f"clips/{name}",
            "start_seconds": round(start / rate, 9),
            "end_seconds_exclusive": round(end / rate, 9),
            "duration_seconds": round((end - start) / rate, 9),
            "start_frame": start,
            "end_frame_exclusive": end,
            "frame_count": end - start,
            "is_full_1s_window": end - start == rate,
            "machine_state": event["machine_state"],
            "machine_state_label": event["machine_state_label"],
            "quality": event["quality"],
            "quality_label": "Conforming",
            "annotation_id": event["id"],
            "annotation_timestamp_seconds": event["timestamp_seconds"],
            "source_recording_id": original_source,
            "source_sample_folder": source.name,
        })

    # Verify the saved clips against the source PCM, including the short tail.
    assert len(rows) == 62
    assert sum(r["is_full_1s_window"] for r in rows) == 61
    assert rows[-1]["frame_count"] == 40572
    assert sum(r["frame_count"] for r in rows) == frames
    cursor = 0
    for row in rows:
        assert row["start_frame"] == cursor
        with wave.open(str(output / row["wav_file"]), "rb") as saved:
            assert (saved.getframerate(), saved.getnchannels(), saved.getsampwidth()) == (rate, channels, width)
            assert saved.getnframes() == row["frame_count"]
            saved_pcm = saved.readframes(saved.getnframes())
        expected_pcm = pcm[row["start_frame"] * frame_bytes:row["end_frame_exclusive"] * frame_bytes]
        assert saved_pcm == expected_pcm, f"PCM mismatch in {row['wav_file']}"
        cursor = row["end_frame_exclusive"]
    assert cursor == frames

    with (output / "clip_labels.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "clip_labels.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        "source_wav": os.path.relpath(audio_path, output),
        "source_events": os.path.relpath(source/'events.json', output),
        "source_sample_folder": source.name,
        "source_recording_id": original_source,
        "time_reference": events["time_reference"],
        "playback_audio_offset_ms": playback["AudioOffsetMs"],
        "sample_rate_hz": rate,
        "channels": channels,
        "bits_per_sample": width * 8,
        "source_frames": frames,
        "source_duration_seconds": frames / rate,
        "window_seconds": 1,
        "hop_seconds": 1,
        "clip_count": len(rows),
        "full_1s_clip_count": 61,
        "partial_clip_count": 1,
        "partial_clip_duration_seconds": rows[-1]["duration_seconds"],
        "partial_clip_policy": "Preserved without padding; distinguish from full one-second training windows.",
        "machine_state": event["machine_state"],
        "quality": event["quality"],
        "quality_label": "Conforming",
        "label_origin": "Existing manual event annotation; not model predictions or independent ratings for each clip.",
        "verification": {"all_saved_pcm_matches_source": True, "covers_all_frames_once": True, "resampled": False, "normalized": False, "padded": False},
    }
    (output / "demo_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "README.txt").write_text(
        "WAV切片与事件标签演示\n"
        "====================\n\n"
        "源样本为‘开腔加工开始’，WAV长61.92秒，44100 Hz、双通道、16位PCM。\n"
        "按1秒、不重叠切片，得到61个完整1秒片段和1个0.92秒尾段，共62段。\n"
        "所有采样帧均保留一次，未重采样、未归一化、未补零。\n\n"
        "clips/ 保存实际WAV切片。\n"
        "clip_labels.csv 与 clip_labels.json 保存每段的时间、采样帧区间、状态及质量标签。\n"
        "demo_summary.json 保存切片参数和验证结果。\n"
        "生成脚本位于项目的Dataset/Preprocessing/demo_windows.py；已有clips目录时会停止，避免覆盖。\n\n"
        "标签对应方式\n"
        "源events.json只有一条0秒标注，明确设置machine_state=machining、quality=3。\n"
        "playback.json中的AudioOffsetMs=0，当前片段播放时间可直接对应WAV时间。\n"
        "因此所有62个切片均继承machining（加工）和3（合格）。这些是既有人工标签，\n"
        "不是模型预测，也不是对62段分别做出的独立质量评价。\n"
        "时间区间采用[start, end)，即包含起点、不包含终点。\n\n"
        "尾段clip_0061.wav对应61.00至61.92秒，只有40572帧。\n"
        "若后续模型要求固定44100帧输入，需明确排除尾段或另行处理；本演示保留原长度。\n"
        "1秒是便于理解的演示窗长，不是既有2048点VAE模型的默认输入长度。\n"
        "全部片段来自同一源录制20261022_174730，不能随机分到训练和测试中当作独立来源。\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
