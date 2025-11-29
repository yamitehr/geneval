"""
Task 6: Constraint Satisfaction
Tests LLM ability to find entities matching multiple simultaneous constraints.
"""

from .task_base import TaskGenerator
from typing import List, Dict, Optional
import random


class ConstraintSatisfactionTask(TaskGenerator):
    """
    Generate questions requiring multiple simultaneous constraints to be satisfied.
    Examples:
    - "Find someone who: (a) is a child of X, (b) lived past age 60, (c) married into House Tudor"
    - "Who among [A,B,C,D,E] satisfies ALL of: NOT a monarch, had siblings, lived in 17th century?"

    Difficulty levels:
    - Easy: 2 constraints
    - Medium: 3 constraints
    - Hard: 4+ constraints
    """

    def __init__(self, data_loader):
        super().__init__(
            data_loader=data_loader,
            task_name="Constraint Satisfaction",
            task_type="constraint_satisfaction"
        )

    def generate_questions(self, count: int, difficulty: str = 'mixed') -> List[Dict]:
        """Generate constraint satisfaction questions"""
        questions = []
        seen_questions = set()

        # Determine number of constraints by difficulty
        if difficulty == 'easy':
            constraint_counts = [2] * count
        elif difficulty == 'medium':
            constraint_counts = [3] * count
        elif difficulty == 'hard':
            constraint_counts = [4, 4, 5] * (count // 3) + [4] * (count % 3)
        else:  # mixed - equal distribution (33/33/33)
            easy_count = count // 3
            medium_count = count // 3
            hard_count = count - easy_count - medium_count
            constraint_counts = [2] * easy_count + [3] * medium_count + [4] * hard_count

        random.shuffle(constraint_counts)

        attempts = 0
        max_attempts = count * 150  # Higher attempts - complex generation

        while len(questions) < count and attempts < max_attempts:
            attempts += 1
            num_constraints = constraint_counts[len(questions)]

            result = self.generate_constraint_question(num_constraints)

            if result:
                question_text, answer_text, diff, metadata = result

                # Avoid duplicates
                if question_text in seen_questions:
                    continue
                seen_questions.add(question_text)

                questions.append(self.create_question_dict(
                    question=question_text,
                    answer=answer_text,
                    difficulty=diff,
                    metadata=metadata
                ))

        return questions

    def generate_constraint_question(self, num_constraints: int) -> Optional[tuple]:
        """Generate a question with N simultaneous constraints"""

        # Available constraint types
        constraint_types = [
            'is_child_of',
            'not_monarch',
            'lived_past_age',
            'married_into_house',
            'lived_in_century',
            'has_siblings',
            'parent_was_monarch',
            'not_born_before_year'
        ]

        # Pick N random constraint types (without replacement)
        if num_constraints > len(constraint_types):
            num_constraints = len(constraint_types)

        selected_constraints = random.sample(constraint_types, num_constraints)

        # Build constraints and find candidates
        constraints_desc = []
        candidates = set(self.data_loader.people.keys())  # Start with all people

        for constraint_type in selected_constraints:
            constraint_desc, passing_people = self.apply_constraint(constraint_type)

            if constraint_desc and passing_people:
                constraints_desc.append(constraint_desc)
                candidates &= passing_people  # Intersection
            else:
                return None  # Constraint generation failed

        # Need at least 1 person who satisfies all constraints
        if len(candidates) < 1:
            return None

        # Pick the correct answer
        correct_answer = random.choice(list(candidates))

        # Create distractors - people who fail at least one constraint
        all_people = set(self.data_loader.people.keys()) - {correct_answer}
        distractors = random.sample(list(all_people), min(4, len(all_people)))

        options = [correct_answer] + distractors
        random.shuffle(options)

        # Format question
        constraints_text = ", ".join(f"({chr(97 + i)}) {c}" for i, c in enumerate(constraints_desc))
        question = f"Who among {', '.join(options)} satisfies ALL of the following: {constraints_text}?"

        # Determine difficulty
        if num_constraints == 2:
            diff = 'easy'
        elif num_constraints == 3:
            diff = 'medium'
        else:
            diff = 'hard'

        metadata = {
            'num_constraints': num_constraints,
            'constraints': selected_constraints,
            'options': options,
            'correct_answer': correct_answer
        }

        return (question, correct_answer, diff, metadata)

    def apply_constraint(self, constraint_type: str) -> tuple:
        """
        Apply a constraint and return (description, set of people who satisfy it)
        """
        if constraint_type == 'is_child_of':
            # Pick a random parent
            parent = self.pick_random_person(
                filter_fn=lambda p: len(self.data_loader.get_children(p['canonical_name'])) >= 1
            )
            if not parent:
                return None, None

            children = set(self.data_loader.get_children(parent))
            return (f"is a child of {parent}", children)

        elif constraint_type == 'not_monarch':
            non_monarchs = {
                p['canonical_name'] for p in self.data_loader.people.values()
                if not p.get('reign_start')
            }
            return ("did NOT become monarch", non_monarchs)

        elif constraint_type == 'lived_past_age':
            age_threshold = random.choice([50, 60, 70])
            lived_past_age = set()

            for person_name, person_data in self.data_loader.people.items():
                birth = self.data_loader.parse_date(person_data['date_of_birth'])
                death = self.data_loader.parse_date(person_data['date_of_death'])

                if birth and death:
                    age = death.year - birth.year
                    if age >= age_threshold:
                        lived_past_age.add(person_name)

            if not lived_past_age:
                return None, None

            return (f"lived past age {age_threshold}", lived_past_age)

        elif constraint_type == 'married_into_house':
            # Pick a royal house
            houses = set()
            for p in self.data_loader.people.values():
                if p.get('house'):
                    houses.add(p['house'])

            if not houses:
                return None, None

            house = random.choice(list(houses))

            married_into_house = set()
            for person_name, person_data in self.data_loader.people.items():
                if person_data.get('spouse'):
                    spouse_name = person_data['spouse']
                    spouse = self.data_loader.find_person(spouse_name)
                    if spouse:
                        spouse_data = self.data_loader.people.get(spouse)
                        if spouse_data and spouse_data.get('house') == house:
                            married_into_house.add(person_name)

            if not married_into_house:
                return None, None

            return (f"married someone from {house}", married_into_house)

        elif constraint_type == 'lived_in_century':
            century_choices = [
                (1100, 1199, "12th century"),
                (1200, 1299, "13th century"),
                (1300, 1399, "14th century"),
                (1400, 1499, "15th century"),
                (1500, 1599, "16th century"),
                (1600, 1699, "17th century"),
            ]

            start_year, end_year, century_name = random.choice(century_choices)

            lived_in_century = set()
            for person_name, person_data in self.data_loader.people.items():
                birth = self.data_loader.parse_date(person_data['date_of_birth'])
                death = self.data_loader.parse_date(person_data['date_of_death'])

                if birth and birth.year <= end_year:
                    if not death or death.year >= start_year:
                        lived_in_century.add(person_name)

            if not lived_in_century:
                return None, None

            return (f"lived during the {century_name}", lived_in_century)

        elif constraint_type == 'has_siblings':
            has_siblings = {
                person_name for person_name in self.data_loader.people.keys()
                if len(self.data_loader.get_siblings(person_name)) >= 1
            }

            if not has_siblings:
                return None, None

            return ("has at least one sibling", has_siblings)

        elif constraint_type == 'parent_was_monarch':
            parent_was_monarch = set()

            for person_name in self.data_loader.people.keys():
                mother, father = self.data_loader.get_parents(person_name)

                for parent in [mother, father]:
                    if parent:
                        parent_data = self.data_loader.people.get(parent)
                        if parent_data and parent_data.get('reign_start'):
                            parent_was_monarch.add(person_name)
                            break

            if not parent_was_monarch:
                return None, None

            return ("has a parent who was monarch", parent_was_monarch)

        elif constraint_type == 'not_born_before_year':
            year_threshold = random.choice([1400, 1500, 1600, 1700])

            not_born_before = set()
            for person_name, person_data in self.data_loader.people.items():
                birth = self.data_loader.parse_date(person_data['date_of_birth'])
                if birth and birth.year >= year_threshold:
                    not_born_before.add(person_name)

            if not not_born_before:
                return None, None

            return (f"was NOT born before {year_threshold}", not_born_before)

        return None, None


if __name__ == '__main__':
    # Test the task generator
    from .data_loader import RoyalFamilyDataLoader

    print("Testing Task 8: Constraint Satisfaction")
    print("="*80)

    loader = RoyalFamilyDataLoader()
    task = ConstraintSatisfactionTask(loader)

    # Generate sample questions
    print("\nGenerating 10 sample questions (mixed difficulty)...")
    questions = task.generate_questions(count=10, difficulty='mixed')

    for i, q in enumerate(questions, 1):
        print(f"\n{i}. [{q['difficulty'].upper()}] ({q['metadata']['num_constraints']} constraints)")
        print(f"   Q: {q['question'][:150]}...")
        print(f"   A: {q['answer']}")

    print(f"\n✓ Successfully generated {len(questions)} questions")
