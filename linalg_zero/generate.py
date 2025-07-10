"""Simple dataset generation script."""


def main() -> None:  # pragma: no cover
    from linalg_zero.generator.core import DatasetGenerator, print_dataset
    from linalg_zero.generator.models import Question
    from linalg_zero.generator.registry import create_default_registry

    print("Linear Algebra Dataset Generator")

    # Show available topics
    registry = create_default_registry()
    print("Available topics:", registry.list_topics())

    # Generate and display basic dataset
    print("\n=== Basic Dataset ===")

    def matrix_only_validator(question: Question) -> bool:
        return "matrix" in question.text.lower() and len(question.answer) > 0

    generator = DatasetGenerator(topic="linear_algebra", validator_factory=matrix_only_validator)
    questions = generator.generate_dataset(num_questions=3)
    print_dataset(questions)

    # Generate arithmetic dataset
    print("\n=== Arithmetic Dataset ===")
    arithmetic_generator = DatasetGenerator(topic="arithmetic")
    arithmetic_questions = arithmetic_generator.generate_dataset(num_questions=2)
    print_dataset(arithmetic_questions)


if __name__ == "__main__":  # pragma: no cover
    main()
