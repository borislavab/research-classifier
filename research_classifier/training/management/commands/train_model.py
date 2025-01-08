from django.core.management.base import BaseCommand
from django.conf import settings
from research_classifier.training.trainer import get_trainer, train_and_save


class Command(BaseCommand):
    help = "Train the research classifier model"

    def add_arguments(self, parser):
        parser.add_argument(
            "--dataset", type=str, help="Path to the dataset", default=None
        )
        parser.add_argument(
            "--output-dir",
            type=str,
            help="Directory to save model outputs",
            default="./output",
        )
        parser.add_argument(
            "--epochs", type=int, help="Number of training epochs", default=3
        )
        parser.add_argument(
            "--sample-count",
            type=int,
            help="Number of samples to use (for testing) - if None, all samples are used",
            default=None,
        )
        parser.add_argument(
            "--from-scratch",
            type=bool,
            help="If True, the model will be trained from scratch. Otherwise, it resumes from checkpoint which should be present in output directory",
            default=False,
        )

    def handle(self, *args, **options):
        self.stdout.write("Starting model training...")

        trainer = get_trainer(
            dataset_path=options["dataset"],
            output_dir=options["output_dir"],
            num_epochs=options["epochs"],
            sample_head=options["sample_count"],
        )

        train_and_save(trainer)

        self.stdout.write(self.style.SUCCESS("Model training completed successfully!"))
