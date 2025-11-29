"""
GENEVAL: Genealogical Reasoning Evaluation Benchmark
Main script to generate the complete dataset of 7 diverse reasoning tasks.

Usage:
    python generate_data.py [--count TOTAL_QUESTIONS] [--output OUTPUT_FILE]

Example:
    python generate_data.py --count 220 --output data/examples.jsonl
"""

import json
import argparse
from collections import defaultdict

# Import data loader and task generators from tasks package
from tasks import (
    RoyalFamilyDataLoader,
    MultiHopTraversalTask,
    TemporalReasoningTask,
    ComparativeLifespanTask,
    NegativeReasoningTask,
    SiblingInferenceTask
)

# Additional task types
from tasks.task_6_constraint_satisfaction import ConstraintSatisfactionTask
from tasks.task_7_adversarial_ambiguity import AdversarialAmbiguityTask


def generate_dataset(total_count=220, output_file='data/examples.jsonl'):
    """
    Generate the complete GENEVAL dataset with diverse reasoning tasks.

    Args:
        total_count: Total number of questions to generate (default: 220)
        output_file: Path to save the generated dataset (default: data/examples.jsonl)

    Returns:
        List of generated questions
    """
    print("="*80)
    print("GENEVAL: Genealogical Reasoning Evaluation Benchmark")
    print("Generating comprehensive multi-task dataset")
    print("="*80)

    # Initialize data loader
    print("\n[1/4] Loading British Royal Family data...")
    loader = RoyalFamilyDataLoader()

    # Initialize all task generators
    print("\n[2/4] Initializing task generators...")

    tasks = [
        (MultiHopTraversalTask(loader), 99),           # Multi-hop reasoning (20%)
        (TemporalReasoningTask(loader), 72),           # Temporal reasoning (14%)
        (NegativeReasoningTask(loader), 72),           # Negative reasoning (14%)
        (SiblingInferenceTask(loader), 72),            # Sibling inference (14%)
        (ConstraintSatisfactionTask(loader), 63),      # Constraint satisfaction (13%)
        (AdversarialAmbiguityTask(loader), 63),        # Adversarial ambiguity (13%)
        (ComparativeLifespanTask(loader), 60),         # Lifespan comparison (12%)
    ]

    total_allocated = sum(count for _, count in tasks)
    print(f"\n✓ Total questions allocated: {total_allocated}")

    print(f"✓ Initialized {len(tasks)} task generators")
    for task, count in tasks:
        print(f"  - {task.task_name}: {count} questions")

    # Generate questions from each task
    print(f"\n[3/4] Generating {total_count} questions across all tasks...")
    all_questions = []

    for task, count in tasks:
        print(f"\n  Generating {task.task_name}...")
        questions = task.generate_questions(count=count, difficulty='mixed')
        all_questions.extend(questions)
        print(f"  ✓ Generated {len(questions)} questions")

    # Shuffle to mix tasks
    import random
    random.shuffle(all_questions)

    # Save to JSONL file
    print(f"\n[4/4] Saving dataset to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for question in all_questions:
            f.write(json.dumps(question, ensure_ascii=False) + '\n')

    print(f"✓ Saved {len(all_questions)} questions to {output_file}")

    # Print statistics
    print("\n" + "="*80)
    print("DATASET STATISTICS")
    print("="*80)

    # Count by task type
    task_counts = defaultdict(int)
    for q in all_questions:
        task_counts[q['task_name']] += 1

    print(f"\nTotal questions: {len(all_questions)}")
    print(f"\nQuestions by task type:")
    for task_name, count in sorted(task_counts.items()):
        percentage = (count / len(all_questions)) * 100
        print(f"  {task_name:.<45} {count:>3} ({percentage:>5.1f}%)")

    # Count by difficulty
    difficulty_counts = defaultdict(int)
    for q in all_questions:
        difficulty_counts[q['difficulty']] += 1

    print(f"\nQuestions by difficulty:")
    for difficulty in ['easy', 'medium', 'hard']:
        count = difficulty_counts[difficulty]
        percentage = (count / len(all_questions)) * 100 if len(all_questions) > 0 else 0
        print(f"  {difficulty.capitalize():.<45} {count:>3} ({percentage:>5.1f}%)")

    # Sample questions
    print(f"\n" + "="*80)
    print("SAMPLE QUESTIONS (3 per task type)")
    print("="*80)

    for task_name in sorted(task_counts.keys()):
        samples = [q for q in all_questions if q['task_name'] == task_name][:3]
        print(f"\n{task_name}:")
        for i, q in enumerate(samples, 1):
            print(f"\n  {i}. [{q['difficulty'].upper()}]")
            print(f"     Q: {q['question']}")
            print(f"     A: {q['answer']}")

    print("\n" + "="*80)
    print("✓ Dataset generation complete!")
    print("="*80)

    return all_questions


def main():
    """Main entry point for command-line usage"""
    parser = argparse.ArgumentParser(
        description='Generate GENEVAL dataset for testing LLM genealogical reasoning'
    )
    parser.add_argument(
        '--count',
        type=int,
        default=220,
        help='Total number of questions to generate (default: 220)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/examples.jsonl',
        help='Output file path (default: data/examples.jsonl)'
    )

    args = parser.parse_args()

    # Generate dataset
    questions = generate_dataset(
        total_count=args.count,
        output_file=args.output
    )

    print(f"\n✓ Successfully generated {len(questions)} questions")
    print(f"✓ Dataset saved to: {args.output}")
    print(f"\nNext steps:")
    print(f"  1. Review the generated questions in {args.output}")
    print(f"  2. Run evaluation: python run_eval.py")


if __name__ == '__main__':
    main()
