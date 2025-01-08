from django.core.management.base import BaseCommand
from research_classifier.preprocessing.oversampling import (
    oversample_llama,
    store_samples,
)
from research_classifier.preprocessing.dataset import load


class Command(BaseCommand):
    help = "Oversample a given number of samples for specific labels in the dataset using Llama 3.2 model. Prints the prompts used to generate the samples. Stores results in jsonl format in the specified output path."

    def add_arguments(self, parser):
        parser.add_argument("dataset_path", type=str, help="Path to the input dataset")
        parser.add_argument(
            "output_path", type=str, help="Path to store generated samples"
        )
        parser.add_argument(
            "target_count",
            type=int,
            help="Number of samples to generate for each label",
        )
        parser.add_argument(
            "labels", nargs="+", type=int, help="Label to oversample (indices)"
        )

    def handle(self, *args, **options):
        dataset = load(dataset_path=options["dataset_path"])
        generated_samples = oversample_llama(
            dataset, labels=options["labels"], target_count=options["target_count"]
        )
        store_samples(generated_samples, options["output_path"])

        self.stdout.write(
            self.style.SUCCESS(
                f"Successfully generated samples in {options['output_path']}"
            )
        )
