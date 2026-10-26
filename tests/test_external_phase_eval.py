from __future__ import annotations

from phase_classification.workflows.external_phase_eval import (
    _eval_phase_id,
    _phase_for_dataset,
    build_parser,
    discover_autolaparo,
    discover_cholec80,
    discover_m2cai16,
    sample_annotation,
)


def test_cholec80_adapter_discovers_test_videos_only(tmp_path):
    videos = tmp_path / "videos"
    annotations = tmp_path / "phase_annotations"
    videos.mkdir()
    annotations.mkdir()
    for number in (1, 41):
        (videos / f"video{number:02d}.mp4").touch()
        (annotations / f"video{number:02d}-phase.txt").write_text(
            "Frame\tPhase\n0\tPreparation\n", encoding="utf-8"
        )

    discovered = discover_cholec80(tmp_path, "test")

    assert [item.video_id for item in discovered] == ["video41"]


def test_cholec80_dense_phase_text_samples_at_requested_times(tmp_path):
    annotation = tmp_path / "video01-phase.txt"
    annotation.write_text(
        "Frame\tPhase\n0\tPreparation\n25\tCalot Triangle Dissection\n",
        encoding="utf-8",
    )

    assert sample_annotation(
        annotation, "cholec80", [0.0, 1.0], annotation_fps=25
    ) == [0, 1]
    assert _phase_for_dataset("1", "cholec80") == 1


def test_m2cai_trocar_and_preparation_collapse_to_the_same_model_class():
    assert _phase_for_dataset("TrocarPlacement", "m2cai16") == 0
    assert _phase_for_dataset("Preparation", "m2cai16") == 1
    assert _eval_phase_id(0, "m2cai16") == 0
    assert _eval_phase_id(1, "m2cai16") == 0
    assert _eval_phase_id(2, "m2cai16") == 1


def test_m2cai16_adapter_reads_camma_sidecar_annotations(tmp_path):
    videos = tmp_path / "videos"
    annotations = tmp_path / "annotations"
    videos.mkdir()
    annotations.mkdir()
    (videos / "workflow_video_01.mp4").touch()
    (annotations / "workflow_video_01.txt").write_text(
        "Frame\tPhase\n0\tTrocarPlacement\n25\tPreparation\n",
        encoding="utf-8",
    )

    discovered = discover_m2cai16(tmp_path, "train")
    assert [item.video_id for item in discovered] == ["workflow_video_01"]
    assert sample_annotation(
        discovered[0].annotation_path, "m2cai16", [0.0, 1.0], annotation_fps=25
    ) == [0, 1]


def test_autolaparo_adapter_pairs_task1_video_with_sidecar(tmp_path):
    videos = tmp_path / "video_src_path"
    annotations = tmp_path / "annotations"
    videos.mkdir()
    annotations.mkdir()
    (videos / "task1_v01.mp4").touch()
    (annotations / "task1_v01.csv").write_text(
        "frame,phase\n0,Preparation\n1,Suturing\n", encoding="utf-8"
    )

    discovered = discover_autolaparo(tmp_path)
    assert [item.video_id for item in discovered] == ["task1_v01"]
    assert sample_annotation(
        discovered[0].annotation_path, "autolaparo", [0.0, 1.0], annotation_fps=1
    ) == [0, 5]


def test_autolaparo_taxonomy_is_not_mapped_to_cholec_classes():
    assert _phase_for_dataset("Suturing", "autolaparo") == 5
    assert _eval_phase_id(5, "autolaparo") is None


def test_external_eval_cli_accepts_a_single_video_id(tmp_path):
    args = build_parser().parse_args(
        [
            "cholec80",
            "--root",
            str(tmp_path),
            "--checkpoint",
            str(tmp_path / "best.pt"),
            "--video",
            "video41",
        ]
    )
    assert args.video == "video41"
