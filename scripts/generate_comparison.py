"""
Generate all final comparison results.
"""

import argparse
parser = argparse.ArgumentParser()

from utils.comparison_plotter import ComparisonPlotter
from utils.table_generator import TableGenerator
from utils.report_generator import ReportGenerator

def main():

    parser.add_argument(
        "--dataset",
        required=True,
        choices=["cifar10", "stl10"],
    )

    args = parser.parse_args()

    dataset_name = args.dataset

    print("=" * 60)
    print("Generating Final Dissertation Results")
    print("=" * 60)

    ComparisonPlotter(dataset_name).generate_all()
    TableGenerator(dataset_name).generate_all()
    ReportGenerator(dataset_name).generate()

    print("=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()