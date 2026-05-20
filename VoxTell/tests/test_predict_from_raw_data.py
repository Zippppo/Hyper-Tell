import numpy as np
import pytest

from voxtell.inference import predict_from_raw_data as cli


def test_get_reader_writer_accepts_only_nifti_paths():
    assert cli.get_reader_writer("case001.nii").__class__.__name__ == "NibabelIOWithReorient"
    assert cli.get_reader_writer("case001.nii.gz").__class__.__name__ == "NibabelIOWithReorient"

    with pytest.raises(ValueError, match="Unsupported file format"):
        cli.get_reader_writer("case001.txt.gz")


def test_save_segmentation_sanitizes_prompt_names_and_uses_input_suffix(tmp_path, monkeypatch):
    calls = []

    class FakeWriter:
        def write_seg(self, segmentation, output_file, properties):
            calls.append((segmentation.copy(), output_file, properties))

    monkeypatch.setattr(cli, "NibabelIOWithReorient", FakeWriter)
    segmentation = np.ones((2, 2, 2), dtype=np.uint8)
    properties = {"spacing": (1.0, 1.0, 1.0)}

    cli.save_segmentation(
        segmentation=segmentation,
        output_folder=tmp_path,
        input_filename="case001",
        properties=properties,
        prompt_name="right kidney/lower pole",
        suffix=".nii.gz",
    )

    assert len(calls) == 1
    saved_segmentation, output_file, saved_properties = calls[0]
    assert np.array_equal(saved_segmentation, segmentation)
    assert output_file == str(tmp_path / "case001_right_kidney_lower_pole.nii.gz")
    assert saved_properties is properties
