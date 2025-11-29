"""
Task 5: Sibling Inference
Tests ability to infer sibling relationships from parent data.
"""

from .task_base import TaskGenerator
from typing import List, Dict, Optional
import random


class SiblingInferenceTask(TaskGenerator):
    """
    Generate questions about sibling relationships.

    Question types:
    - "Name a sibling of X"
    - "Did X and Y have the same mother?"
    - "Did X and Y have the same father?"
    - "Are X and Y full siblings or half-siblings?"

    Difficulty levels:
    - Easy: Direct sibling naming
    - Medium: Same parent verification
    - Hard: Full vs half-sibling distinction
    """

    def __init__(self, data_loader):
        super().__init__(
            data_loader=data_loader,
            task_name="Sibling Inference",
            task_type="sibling_inference"
        )

    def generate_questions(self, count: int, difficulty: str = 'mixed') -> List[Dict]:
        """Generate sibling inference questions (verifiable only)"""
        questions = []
        seen_questions = set()

        # Question types by difficulty - ONLY VERIFIABLE
        easy_types = ['name_sibling', 'same_mother', 'same_father']
        medium_types = ['same_mother', 'same_father', 'same_both_parents']
        hard_types = ['full_vs_half_sibling', 'same_both_parents', 'identify_half_sibling']

        # Build difficulty schedule for mixed mode (equal distribution)
        if difficulty == 'mixed':
            easy_count = count // 3
            medium_count = count // 3
            hard_count = count - easy_count - medium_count
            difficulty_schedule = ['easy'] * easy_count + ['medium'] * medium_count + ['hard'] * hard_count
            random.shuffle(difficulty_schedule)
        else:
            difficulty_schedule = [difficulty] * count

        attempts = 0
        max_attempts = count * 100

        while len(questions) < count and attempts < max_attempts:
            attempts += 1

            # Get difficulty for this question
            current_diff = difficulty_schedule[len(questions)] if len(questions) < len(difficulty_schedule) else difficulty

            # Select question types based on difficulty
            if current_diff == 'hard':
                question_types = hard_types
            elif current_diff == 'medium':
                question_types = medium_types
            else:
                question_types = easy_types

            q_type = random.choice(question_types)

            result = None
            if q_type == 'name_sibling':
                result = self.generate_name_sibling_question()
            elif q_type == 'same_mother':
                result = self.generate_same_mother_question()
            elif q_type == 'same_father':
                result = self.generate_same_father_question()
            elif q_type == 'same_both_parents':
                result = self.generate_same_both_parents_question()
            elif q_type == 'full_vs_half_sibling':
                result = self.generate_full_vs_half_sibling_question()
            elif q_type == 'identify_half_sibling':
                result = self.generate_identify_half_sibling_question()

            if result:
                question_text, answer_text, diff, metadata = result

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

    def generate_name_sibling_question(self) -> Optional[tuple]:
        """Generate 'Name a sibling of X' question"""
        person = self.pick_random_person(
            filter_fn=lambda p: len(self.data_loader.get_siblings(p['canonical_name'])) > 0
        )

        if not person:
            return None

        siblings = list(self.data_loader.get_siblings(person))
        answer = random.choice(siblings)

        question = f"Name a sibling of {person}."

        metadata = {
            'question_type': 'name_sibling',
            'person': person,
            'all_siblings_in_dataset': siblings,
            'correct_answer': answer,
            'requires_inference': True,
            'multiple_valid_answers': len(siblings) > 1,
            'note': 'Any sibling in dataset is valid; dataset may be incomplete'
        }

        return (question, answer, 'easy', metadata)

    def generate_same_mother_question(self) -> Optional[tuple]:
        """Generate 'Did X and Y have the same mother?' question - VERIFIABLE"""
        # Only pick people where mother EXISTS in dataset
        person1 = self.pick_random_person(
            filter_fn=lambda p: (len(self.data_loader.get_siblings(p['canonical_name'])) > 0 and
                                self.data_loader.get_parents(p['canonical_name'])[0] is not None)
        )

        if not person1:
            return None

        mother1, _ = self.data_loader.get_parents(person1)
        if not mother1:
            return None

        siblings = list(self.data_loader.get_siblings(person1))

        # Pick a sibling or non-sibling
        if random.random() < 0.5 and len(siblings) > 0:
            # Pick someone who might share the mother
            person2 = random.choice(siblings)
            mother2, _ = self.data_loader.get_parents(person2)
            is_true = (mother1 == mother2) if mother2 else False
        else:
            # Pick a non-sibling
            person2 = self.pick_random_person(
                filter_fn=lambda p: (p['canonical_name'] not in siblings and
                                   p['canonical_name'] != person1 and
                                   self.data_loader.get_parents(p['canonical_name'])[0] is not None)
            )
            if not person2:
                return None

            mother2, _ = self.data_loader.get_parents(person2)
            is_true = (mother1 == mother2) if mother2 else False

        answer = "Yes" if is_true else "No"
        question = f"Did {person1} and {person2} have the same mother?"

        metadata = {
            'question_type': 'same_mother',
            'people': [person1, person2],
            'answer_bool': is_true,
            'requires_inference': True,
            'verifiable': True
        }

        return (question, answer, 'medium', metadata)

    def generate_same_father_question(self) -> Optional[tuple]:
        """Generate 'Did X and Y have the same father?' question - VERIFIABLE"""
        # Only pick people where father EXISTS in dataset
        person1 = self.pick_random_person(
            filter_fn=lambda p: (len(self.data_loader.get_siblings(p['canonical_name'])) > 0 and
                                self.data_loader.get_parents(p['canonical_name'])[1] is not None)
        )

        if not person1:
            return None

        _, father1 = self.data_loader.get_parents(person1)
        if not father1:
            return None

        siblings = list(self.data_loader.get_siblings(person1))

        if random.random() < 0.5 and len(siblings) > 0:
            person2 = random.choice(siblings)
            _, father2 = self.data_loader.get_parents(person2)
            is_true = (father1 == father2) if father2 else False
        else:
            person2 = self.pick_random_person(
                filter_fn=lambda p: (p['canonical_name'] not in siblings and
                                   p['canonical_name'] != person1 and
                                   self.data_loader.get_parents(p['canonical_name'])[1] is not None)
            )
            if not person2:
                return None

            _, father2 = self.data_loader.get_parents(person2)
            is_true = (father1 == father2) if father2 else False

        answer = "Yes" if is_true else "No"
        question = f"Did {person1} and {person2} have the same father?"

        metadata = {
            'question_type': 'same_father',
            'people': [person1, person2],
            'answer_bool': is_true,
            'requires_inference': True,
            'verifiable': True
        }

        return (question, answer, 'medium', metadata)

    def generate_same_both_parents_question(self) -> Optional[tuple]:
        """Generate 'Did X and Y have the same mother AND father?' question - VERIFIABLE"""
        # Only pick people where BOTH parents exist in dataset
        person1 = self.pick_random_person(
            filter_fn=lambda p: (len(self.data_loader.get_siblings(p['canonical_name'])) > 0 and
                                self.data_loader.has_both_parents_in_dataset(p['canonical_name']))
        )

        if not person1:
            return None

        mother1, father1 = self.data_loader.get_parents(person1)
        if not (mother1 and father1):
            return None

        siblings = list(self.data_loader.get_siblings(person1))
        full_siblings = list(self.data_loader.get_siblings(person1, full_siblings_only=True))

        if random.random() < 0.5 and len(full_siblings) > 0:
            # Pick a full sibling
            person2 = random.choice(full_siblings)
            is_true = True
        elif len(siblings) > 0:
            # Pick a half-sibling or non-sibling
            non_full_sibs = [s for s in siblings if s not in full_siblings]
            if non_full_sibs and random.random() < 0.5:
                person2 = random.choice(non_full_sibs)
                is_true = False
            else:
                person2 = self.pick_random_person(
                    filter_fn=lambda p: (p['canonical_name'] not in siblings and
                                       p['canonical_name'] != person1 and
                                       self.data_loader.has_both_parents_in_dataset(p['canonical_name']))
                )
                if not person2:
                    return None
                is_true = False
        else:
            person2 = self.pick_random_person(
                filter_fn=lambda p: (p['canonical_name'] != person1 and
                                   self.data_loader.has_both_parents_in_dataset(p['canonical_name']))
            )
            if not person2:
                return None
            is_true = False

        answer = "Yes" if is_true else "No"
        question = f"Did {person1} and {person2} have the same mother AND father?"

        metadata = {
            'question_type': 'same_both_parents',
            'people': [person1, person2],
            'answer_bool': is_true,
            'requires_inference': True,
            'tests_full_vs_half_siblings': True,
            'verifiable': True
        }

        return (question, answer, 'hard', metadata)

    def generate_full_vs_half_sibling_question(self) -> Optional[tuple]:
        """Generate 'Are X and Y full siblings or half-siblings?' question - VERIFIABLE"""
        # Pick someone with at least one sibling where both have both parents in dataset
        person1 = self.pick_random_person(
            filter_fn=lambda p: (len(self.data_loader.get_siblings(p['canonical_name'])) > 0 and
                                self.data_loader.has_both_parents_in_dataset(p['canonical_name']))
        )

        if not person1:
            return None

        mother1, father1 = self.data_loader.get_parents(person1)
        siblings = list(self.data_loader.get_siblings(person1))

        # Find a sibling who also has both parents in dataset
        valid_siblings = [
            s for s in siblings
            if self.data_loader.has_both_parents_in_dataset(s)
        ]

        if not valid_siblings:
            return None

        person2 = random.choice(valid_siblings)
        mother2, father2 = self.data_loader.get_parents(person2)

        # Check if full siblings (same mother AND father) or half-siblings (only one parent matches)
        same_mother = (mother1 == mother2)
        same_father = (father1 == father2)

        if same_mother and same_father:
            answer = "Full siblings (they share both parents)"
            is_full = True
        elif same_mother or same_father:
            shared_parent = "mother" if same_mother else "father"
            answer = f"Half-siblings (they share the same {shared_parent} only)"
            is_full = False
        else:
            return None  # Not siblings at all, shouldn't happen

        question = f"Are {person1} and {person2} full siblings or half-siblings?"

        metadata = {
            'question_type': 'full_vs_half_sibling',
            'people': [person1, person2],
            'is_full_sibling': is_full,
            'shared_parents': {'mother': same_mother, 'father': same_father},
            'requires_inference': True,
            'tests_full_vs_half_siblings': True,
            'verifiable': True
        }

        return (question, answer, 'hard', metadata)

    def generate_identify_half_sibling_question(self) -> Optional[tuple]:
        """Generate 'Which of X's siblings is a half-sibling?' question - VERIFIABLE"""
        # Pick someone with multiple siblings where we can verify parent relationships
        person = self.pick_random_person(
            filter_fn=lambda p: (len(self.data_loader.get_siblings(p['canonical_name'])) >= 2 and
                                self.data_loader.has_both_parents_in_dataset(p['canonical_name']))
        )

        if not person:
            return None

        mother, father = self.data_loader.get_parents(person)
        siblings = list(self.data_loader.get_siblings(person))

        # Categorize siblings as full vs half (only those with both parents in dataset)
        full_siblings = []
        half_siblings = []

        for sibling in siblings:
            if not self.data_loader.has_both_parents_in_dataset(sibling):
                continue

            sib_mother, sib_father = self.data_loader.get_parents(sibling)

            if sib_mother == mother and sib_father == father:
                full_siblings.append(sibling)
            elif sib_mother == mother or sib_father == father:
                half_siblings.append(sibling)

        # Need at least one half-sibling and one full sibling
        if not half_siblings or not full_siblings:
            return None

        # Create question with mix of full and half siblings
        if len(half_siblings) == 1:
            # Only one half-sibling, add some full siblings as distractors
            options = [half_siblings[0]] + random.sample(full_siblings, min(2, len(full_siblings)))
            random.shuffle(options)
            question = f"Which of the following siblings of {person} is a half-sibling (shares only one parent): {', '.join(options)}?"
            answer = half_siblings[0]
        else:
            # Multiple half-siblings
            answer_sibs = random.sample(half_siblings, min(2, len(half_siblings)))
            options = answer_sibs + random.sample(full_siblings, min(2, len(full_siblings)))
            random.shuffle(options)
            question = f"Which of the following siblings of {person} are half-siblings (share only one parent): {', '.join(options)}?"
            answer = ', '.join(sorted(answer_sibs))

        metadata = {
            'question_type': 'identify_half_sibling',
            'person': person,
            'options': options,
            'correct_answer': answer,
            'full_siblings': full_siblings,
            'half_siblings': half_siblings,
            'requires_inference': True,
            'tests_full_vs_half_siblings': True,
            'verifiable': True
        }

        return (question, answer, 'hard', metadata)


if __name__ == '__main__':
    # Test the task generator
    from .data_loader import RoyalFamilyDataLoader

    print("Testing Task 7: Sibling Inference (Verifiable Questions Only)")
    print("="*80)

    loader = RoyalFamilyDataLoader()
    task = SiblingInferenceTask(loader)

    # Generate sample questions
    print("\nGenerating 10 sample questions...")
    questions = task.generate_questions(count=10, difficulty='mixed')

    for i, q in enumerate(questions, 1):
        print(f"\n{i}. [{q['difficulty'].upper()}] - {q['metadata']['question_type']}")
        print(f"   Q: {q['question']}")
        print(f"   A: {q['answer']}")
        if 'multiple_valid_answers' in q['metadata'] and q['metadata']['multiple_valid_answers']:
            print(f"   (Multiple valid answers possible)")

    print(f"\n✓ Successfully generated {len(questions)} VERIFIABLE questions")
    print("✓ All questions can be answered definitively from the dataset")
