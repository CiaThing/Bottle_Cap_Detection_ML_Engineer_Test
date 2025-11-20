from unittest.mock import MagicMock, patch

import pytest
import yaml

from bsort.main import infer_image, load_config, train_model


# --- Test 1: Load Config ---
def test_load_config_success(tmp_path):
    """Test apakah load_config berhasil membaca file YAML valid."""
    # 1. Buat file dummy setting.yaml di folder temporary
    d = tmp_path / "settings_dummy.yaml"
    data = {"epochs": 5, "model_path": "yolo_test.pt"}

    # Menulis file dummy
    with open(d, "w", encoding="utf-8") as f:
        yaml.dump(data, f)

    # 2. Panggil fungsi
    config = load_config(str(d))

    # 3. Assert (Pastikan isinya benar)
    assert config["epochs"] == 5
    assert config["model_path"] == "yolo_test.pt"


def test_load_config_not_found():
    """Test apakah error muncul jika file tidak ada."""
    with pytest.raises(FileNotFoundError):
        load_config("file_ngawur_yang_tidak_ada.yaml")


# --- Test 2: Train Model (Pakai Mocking) ---
@patch("bsort.main.YOLO")  # Kita mock class YOLO biar tidak download model beneran
def test_train_model(mock_yolo):
    """Test apakah fungsi train memanggil model.train dengan parameter yg benar."""
    # Setup config dummy
    config = {
        "model_path": "yolov8n.pt",
        "dataset_path": "data.yaml",
        "epochs": 1,
        "image_size": 640,
        "batch_size": 2,
    }

    # Panggil fungsi
    train_model(config)

    # Assertions:
    # 1. Pastikan YOLO di-init
    mock_yolo.assert_called_once_with("yolov8n.pt")

    # 2. Ambil instance model yg dimock
    model_instance = mock_yolo.return_value

    # 3. Pastikan method .train() dipanggil
    model_instance.train.assert_called_once()

    # 4. Cek apakah parameter epochs masuk dengan benar
    _, kwargs = model_instance.train.call_args
    assert kwargs["epochs"] == 1
    assert kwargs["batch"] == 2


# --- Test 3: Infer Image (Pakai Mocking) ---
@patch("bsort.main.YOLO")  # Mock YOLO
@patch("os.path.exists")  # Mock os.path.exists (biar gak perlu file gambar asli)
def test_infer_image(mock_exists, mock_yolo):
    """Test inference flow tanpa butuh file gambar asli."""

    # Setup: Anggap semua file (model & gambar) ADA (True)
    mock_exists.return_value = True

    # Setup: Mock return value dari predict (biar gak error pas print results)
    mock_result = MagicMock()
    mock_result.save_dir = "runs/detect/predict"

    # Mock instance model
    model_instance = mock_yolo.return_value
    model_instance.predict.return_value = [mock_result]  # Return list of results

    config = {"model_path": "best.pt", "confidence_threshold": 0.5}

    # Panggil fungsi
    infer_image(config, "dummy_image.jpg")

    # Assertions
    mock_yolo.assert_called_with("best.pt")
    model_instance.predict.assert_called_once()

    # Cek parameter predict
    _, kwargs = model_instance.predict.call_args
    assert kwargs["source"] == "dummy_image.jpg"
    assert kwargs["conf"] == 0.5
