import sys
from research_classifier.training.trainer import evaluate

# file to expose an entrypoint for the evaluation script
if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python evaluate.py <model_path> <output_dir> <dataset_path>")
        sys.exit(1)
    model_path = sys.argv[1]
    output_dir = sys.argv[2]
    dataset_path = sys.argv[3]
    evaluate(
        model_path=model_path,
        output_dir=output_dir,
        dataset_path=dataset_path,
    )
