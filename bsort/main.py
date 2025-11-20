import argparse
import yaml
import os
from typing import Dict, Any
from ultralytics import YOLO


def load_config(config_path: str) -> Dict[str, Any]:
    """Loads configuration from a YAML file.

    Args:
        config_path (str): Path to the .yaml configuration file.

    Returns:
        Dict[str, Any]: Dictionary containing configuration parameters.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def train_model(config: Dict[str, Any]) -> None:
    """Trains the YOLO model based on configuration settings.

    Args:
        config (Dict[str, Any]): Configuration dictionary.
    """
    print(f"🚀 Starting training with model: {config.get('model_path', 'yolov8n.pt')}")

    # Load model (load weights if exist, else load pretrained nano)
    model_path = config.get("model_path", "yolov8n.pt")
    model = YOLO(model_path)

    # Train
    model.train(
        data=config.get("dataset_path"),
        epochs=config.get("epochs", 10),
        imgsz=config.get("image_size", 320),
        batch=config.get("batch_size", 8),
        lr0=config.get("learning_rate", 0.01),
        project="runs/train",
        name="bsort_run",
    )
    print("✅ Training finished.")


def infer_image(config: Dict[str, Any], image_path: str) -> None:
    """Runs inference on a single image using the trained model.

    Args:
        config (Dict[str, Any]): Configuration dictionary.
        image_path (str): Path to the input image.

    Raises:
        FileNotFoundError: If the image or model file does not exist.
    """
    model_path = config["model_path"]
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at: {model_path}. Please train first or check path."
        )

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found at: {image_path}")

    print(f"🔍 Running inference on: {image_path}")
    print(f"🤖 Loading model from: {model_path}")

    # Load Model
    model = YOLO(model_path)

    # Run Inference
    results = model.predict(
        source=image_path,
        conf=config.get("confidence_threshold", 0.25),
        imgsz=config.get("image_size", 320),
        save=True,
        project=config.get("output_dir", "runs/detect"),
        name="predict",
    )

    print(f"✅ Inference complete. Results saved in {results[0].save_dir}")


def main() -> None:
    """Main entry point for the CLI program."""
    parser = argparse.ArgumentParser(description="BSORT: Bottle Cap Sorter CLI Tool")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- Subparser for TRAIN ---
    parser_train = subparsers.add_parser("train", help="Train the model")
    parser_train.add_argument(
        "--config", type=str, required=True, help="Path to settings.yaml"
    )

    # --- Subparser for INFER ---
    parser_infer = subparsers.add_parser("infer", help="Run inference on an image")
    parser_infer.add_argument(
        "--config", type=str, required=True, help="Path to settings.yaml"
    )
    parser_infer.add_argument(
        "--image", type=str, required=True, help="Path to image file"
    )

    args = parser.parse_args()

    if args.command == "train":
        config = load_config(args.config)
        train_model(config)

    elif args.command == "infer":
        config = load_config(args.config)
        infer_image(config, args.image)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
