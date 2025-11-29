"""
Task 1: Multi-Hop Relationship Traversal
Tests compositional reasoning through chained parent/spouse relationships.
"""

from .task_base import TaskGenerator
from typing import List, Dict, Optional
import random


class MultiHopTraversalTask(TaskGenerator):
    """
    Generate questions that chain multiple relationship steps.
    Example: "Who is the mother of the spouse of the father of X?"

    Difficulty levels:
    - Easy: 2 hops
    - Medium: 3-4 hops
    - Hard: 5-6 hops with mixed relationship types (including child)
    """

    def __init__(self, data_loader):
        super().__init__(
            data_loader=data_loader,
            task_name="Multi-Hop Relationship Traversal",
            task_type="multi_hop"
        )

    def generate_questions(self, count: int, difficulty: str = 'mixed') -> List[Dict]:
        """Generate multi-hop relationship traversal questions"""
        questions = []
        seen_questions = set()  # Avoid duplicates

        # Determine difficulty distribution
        if difficulty == 'easy':
            hop_counts = [2] * count
        elif difficulty == 'medium':
            hop_counts = ([3] * (count // 2) + [4] * (count - count // 2))
        elif difficulty == 'hard':
            # Hard: 5-6 hops with mixed relationship types (more achievable)
            hop_counts = ([5] * (count // 2) + [6] * (count - count // 2))
        else:  # mixed - equal distribution (33/33/33)
            easy_count = count // 3
            medium_count = count // 3
            hard_count = count - easy_count - medium_count

            # Build hop counts list
            hop_counts = [2] * easy_count
            # For medium: alternate between 3 and 4 hops
            for i in range(medium_count):
                hop_counts.append(3 if i % 2 == 0 else 4)
            # For hard: alternate between 5 and 6 hops
            for i in range(hard_count):
                hop_counts.append(5 if i % 2 == 0 else 6)

        random.shuffle(hop_counts)

        attempts = 0
        max_attempts = count * 100  # Prevent infinite loop

        while len(questions) < count and attempts < max_attempts:
            attempts += 1
            num_hops = hop_counts[len(questions)]

            # Try to find a valid path
            result = self.find_valid_path(num_hops)

            if result:
                start_person, path, end_person = result

                # Create question key to avoid duplicates
                question_key = (start_person, tuple(path))
                if question_key in seen_questions:
                    continue

                seen_questions.add(question_key)

                # Generate question and answer
                question_text = self.create_question_text(start_person, path)
                answer_text = end_person

                # Determine difficulty
                if num_hops == 2:
                    diff = 'easy'
                elif num_hops in [3, 4]:
                    diff = 'medium'
                else:  # 5-6 hops
                    diff = 'hard'

                # Create metadata
                metadata = {
                    'hops': num_hops,
                    'start_person': start_person,
                    'end_person': end_person,
                    'path': path,
                    'has_spouse_relation': 'spouse' in path,
                    'has_child_relation': 'child' in path
                }

                questions.append(self.create_question_dict(
                    question=question_text,
                    answer=answer_text,
                    difficulty=diff,
                    metadata=metadata
                ))

        return questions

    def find_valid_path(self, num_hops: int) -> Optional[tuple]:
        """
        Find a valid path of specified length through the family tree.

        Returns:
            (start_person, path, end_person) or None if no valid path found
        """
        # For hard questions (5+ hops), include child relationships for more complexity
        include_child = num_hops >= 5

        max_attempts = 2000 if include_child else 1000
        for _ in range(max_attempts):
            # Pick a random starting person
            start_person = self.pick_random_person()
            if not start_person:
                continue

            # Build a random path
            path = []
            prev_relation = None
            for hop_idx in range(num_hops):
                if include_child:
                    # Hard questions include child relations and backtracking patterns
                    rand = random.random()

                    # Add backtracking patterns (e.g., father → mother → father → mother)
                    # 20% chance of backtracking if not first hop
                    if hop_idx > 0 and rand < 0.2 and prev_relation in ['mother', 'father']:
                        # Zigzag: opposite parent relation
                        rel_type = 'father' if prev_relation == 'mother' else 'mother'
                    elif rand < 0.45:  # 45% parent relations (reduced from 50%)
                        rel_type = random.choice(['mother', 'father'])
                    elif rand < 0.65:  # 20% spouse relations (reduced from 20%)
                        rel_type = 'spouse'
                    else:  # 35% child relations (increased from 30%)
                        rel_type = 'child'
                else:
                    # Weight towards parent relations (more common)
                    if random.random() < 0.7:
                        rel_type = random.choice(['mother', 'father'])
                    else:
                        rel_type = 'spouse'

                path.append(rel_type)
                prev_relation = rel_type

            # Try to follow this path
            end_person = self.follow_path(start_person, path)

            if end_person:
                return (start_person, path, end_person)

        return None

    def follow_path(self, start_person: str, path: List[str]) -> Optional[str]:
        """
        Follow a relationship path from a starting person.

        Args:
            start_person: Canonical name of starting person
            path: List of relationship types ['mother', 'father', 'spouse', 'child']

        Returns:
            Canonical name of end person, or None if path is invalid
        """
        current_person = start_person

        for relation in path:
            if relation == 'mother':
                mother, _ = self.data_loader.get_parents(current_person)
                if not mother:
                    return None
                current_person = mother

            elif relation == 'father':
                _, father = self.data_loader.get_parents(current_person)
                if not father:
                    return None
                current_person = father

            elif relation == 'spouse':
                person_data = self.data_loader.people.get(current_person)
                if not person_data or not person_data['spouse']:
                    return None

                spouse_name = person_data['spouse']
                spouse_canonical = self.data_loader.find_person(spouse_name)
                if not spouse_canonical:
                    return None

                current_person = spouse_canonical

            elif relation == 'child':
                # Get children of current person
                children = self.get_children(current_person)
                if not children:
                    return None
                # Pick a random child
                current_person = random.choice(children)

            else:
                return None

        return current_person

    def get_children(self, person: str) -> List[str]:
        """Get all children of a person."""
        children = []
        for canonical_name, person_data in self.data_loader.people.items():
            mother, father = self.data_loader.get_parents(canonical_name)
            if mother == person or father == person:
                children.append(canonical_name)
        return children

    def create_question_text(self, start_person: str, path: List[str]) -> str:
        """
        Create natural language question from a relationship path.

        Args:
            start_person: Starting person
            path: List of relationships

        Returns:
            Question text like "Who is the father of the mother of X?"
        """
        # Build the relationship chain in natural reading order
        # Path is executed left-to-right, but we phrase it right-to-left
        relationship_text = path[0]
        for i in range(1, len(path)):
            relationship_text = path[i] + " of the " + relationship_text

        question = f"Who is the {relationship_text} of {start_person}?"
        return question


if __name__ == '__main__':
    # Test the task generator
    from .data_loader import RoyalFamilyDataLoader

    print("Testing Task 1: Multi-Hop Relationship Traversal")
    print("="*80)

    loader = RoyalFamilyDataLoader()
    task = MultiHopTraversalTask(loader)

    # Generate sample questions
    print("\nGenerating 10 sample questions (mixed difficulty)...")
    questions = task.generate_questions(count=10, difficulty='mixed')

    for i, q in enumerate(questions, 1):
        print(f"\n{i}. [{q['difficulty'].upper()}] ({q['metadata']['hops']} hops)")
        print(f"   Q: {q['question']}")
        print(f"   A: {q['answer']}")
        if q['metadata']['has_spouse_relation']:
            print(f"   (includes spouse relationship)")

    print(f"\n✓ Successfully generated {len(questions)} questions")
