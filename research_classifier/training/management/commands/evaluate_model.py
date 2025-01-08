from django.core.management.base import BaseCommand
from research_classifier.training.trainer import evaluate


class Command(BaseCommand):
    help = "Evaluate the model performance across all labels"

    def add_arguments(self, parser):
        parser.add_argument(
            "--model-path", type=str, help="Path to the model checkpoint to evaluate"
        )
        parser.add_argument(
            "--output-dir",
            type=str,
            help="Directory to save evaluation metrics",
            default="./output",
        )
        parser.add_argument(
            "--dataset", type=str, help="Path to the full dataset", default=None
        )
        parser.add_argument(
            "--sample-count",
            type=int,
            help="Number of samples to use (for testing)",
            default=None,
        )

    def handle(self, *args, **options):
        self.stdout.write("Starting model evaluation...")

        evaluate(
            model_path=options["model_path"],
            output_dir=options["output_dir"],
            dataset_path=options["dataset"],
            sample_head=options["sample_count"],
        )

        self.stdout.write(
            self.style.SUCCESS(
                f'Evaluation completed! Metrics saved to {options["output_dir"]}/metrics.json'
            )
        )
