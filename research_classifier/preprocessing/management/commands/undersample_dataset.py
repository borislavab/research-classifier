from django.core.management.base import BaseCommand
from research_classifier.preprocessing.undersampling import undersample
from research_classifier.training.dataset import load


class Command(BaseCommand):
    help = "Undersample dataset to remove majority class samples until their number is close to a given threshold"

    def add_arguments(self, parser):
        parser.add_argument("dataset_path", type=str, help="Path to the input dataset")
        parser.add_argument(
            "output_path", type=str, help="Path where to save the undersampled dataset"
        )
        parser.add_argument(
            "--threshold",
            type=float,
            help="Optional threshold for undersampling (defaults to dataset median of label counts)",
            default=None,
        )

    def handle(self, *args, **options):
        self.stdout.write("Loading dataset...")
        dataset = load(dataset_path=options["dataset_path"])

        self.stdout.write("Undersampling dataset...")
        undersampled = undersample(dataset=dataset, threshold=options["threshold"])

        self.stdout.write("Saving undersampled dataset...")
        undersampled.to_json(options["output_path"])

        self.stdout.write(
            self.style.SUCCESS(
                f"Undersampling complete! New dataset size: {len(undersampled)} samples\n"
                f'Saved to: {options["output_path"]}'
            )
        )
