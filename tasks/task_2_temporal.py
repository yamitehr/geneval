"""
Task 2: Temporal Reasoning
Tests date comparison, chronological ordering, and temporal logic.
"""

from .task_base import TaskGenerator
from typing import List, Dict, Optional
import random
from datetime import datetime


class TemporalReasoningTask(TaskGenerator):
    """
    Generate questions requiring temporal reasoning.
    Examples:
    - "Who was born first, X or Y?"
    - "Did X die before Y was born?"
    - "At what age did X die?"

    Difficulty levels:
    - Easy: 2-person comparisons, simple date arithmetic
    - Medium: 3-person comparisons, reign period queries
    - Hard: 4-5 person ranking, complex temporal logic
    """

    def __init__(self, data_loader):
        super().__init__(
            data_loader=data_loader,
            task_name="Temporal Reasoning",
            task_type="temporal"
        )

    def generate_questions(self, count: int, difficulty: str = 'mixed') -> List[Dict]:
        """Generate temporal reasoning questions"""
        questions = []
        seen_questions = set()

        # Question types
        question_types = [
            'birth_order',      # Who was born first?
            'death_order',      # Who died first?
            'lifespan_overlap', # Did X die before Y was born?
            'age_at_death',     # At what age did X die?
            'age_at_event',     # How old was X when Y was born?
            'reign_during_birth' # Who reigned when X was born?
        ]

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

            # Pick a question type
            q_type = random.choice(question_types)

            result = None
            if q_type == 'birth_order':
                result = self.generate_birth_order_question(current_diff)
            elif q_type == 'death_order':
                result = self.generate_death_order_question(current_diff)
            elif q_type == 'lifespan_overlap':
                result = self.generate_lifespan_overlap_question()
            elif q_type == 'age_at_death':
                result = self.generate_age_at_death_question()
            elif q_type == 'age_at_event':
                result = self.generate_age_at_event_question()
            elif q_type == 'reign_during_birth':
                result = self.generate_reign_during_birth_question()

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

    def generate_birth_order_question(self, difficulty='easy') -> Optional[tuple]:
        """Generate 'Who was born first?' question (2-5 people depending on difficulty)"""
        # Determine number of people based on difficulty
        if difficulty == 'hard':
            num_people = random.choice([4, 5])
            diff_level = 'hard'
        elif difficulty == 'medium':
            num_people = 3
            diff_level = 'medium'
        else:  # easy or mixed
            num_people = 2
            diff_level = 'easy'

        # Pick people with known birth dates
        people_with_dates = [
            (p['canonical_name'], self.data_loader.parse_date(p['date_of_birth']))
            for p in self.data_loader.people.values()
            if p['date_of_birth'] and p['date_of_birth'] != ''
        ]

        people_with_dates = [(name, date) for name, date in people_with_dates if date is not None]

        if len(people_with_dates) < num_people:
            return None

        # Pick random people
        selected = random.sample(people_with_dates, num_people)
        selected.sort(key=lambda x: x[1])  # Sort by date

        people_names = [name for name, _ in selected]

        if num_people == 2:
            question = f"Who was born first, {people_names[0]} or {people_names[1]}?"
            answer = people_names[0]  # Already sorted, first is earliest
        elif num_people == 3:
            question = f"Who was born first among {people_names[0]}, {people_names[1]}, and {people_names[2]}?"
            answer = people_names[0]
        else:  # 4-5 people
            people_str = ', '.join(people_names[:-1]) + f', and {people_names[-1]}'
            question = f"Rank the following people by birth order (earliest to latest): {people_str}"
            answer = ', '.join(people_names)  # Already sorted

        metadata = {
            'question_type': 'birth_order',
            'people': people_names,
            'num_people': num_people,
            'requires_date_comparison': True,
            'requires_ranking': num_people > 3
        }

        return (question, answer, diff_level, metadata)

    def generate_death_order_question(self, difficulty='easy') -> Optional[tuple]:
        """Generate 'Who died first?' question (2-5 people depending on difficulty)"""
        # Determine number of people based on difficulty
        if difficulty == 'hard':
            num_people = random.choice([4, 5])
            diff_level = 'hard'
        elif difficulty == 'medium':
            num_people = 3
            diff_level = 'medium'
        else:  # easy or mixed
            num_people = 2
            diff_level = 'easy'

        # Pick people with known death dates
        people_with_dates = [
            (p['canonical_name'], self.data_loader.parse_date(p['date_of_death']))
            for p in self.data_loader.people.values()
            if p['date_of_death'] and p['date_of_death'] != ''
        ]

        people_with_dates = [(name, date) for name, date in people_with_dates if date is not None]

        if len(people_with_dates) < num_people:
            return None

        # Pick random people
        selected = random.sample(people_with_dates, num_people)
        selected.sort(key=lambda x: x[1])  # Sort by date

        people_names = [name for name, _ in selected]

        if num_people == 2:
            question = f"Who died first, {people_names[0]} or {people_names[1]}?"
            answer = people_names[0]
        elif num_people == 3:
            question = f"Who died first among {people_names[0]}, {people_names[1]}, and {people_names[2]}?"
            answer = people_names[0]
        else:  # 4-5 people
            people_str = ', '.join(people_names[:-1]) + f', and {people_names[-1]}'
            question = f"Rank the following people by death order (earliest to latest): {people_str}"
            answer = ', '.join(people_names)

        metadata = {
            'question_type': 'death_order',
            'people': people_names,
            'num_people': num_people,
            'requires_date_comparison': True,
            'requires_ranking': num_people > 3
        }

        return (question, answer, diff_level, metadata)

    def generate_lifespan_overlap_question(self) -> Optional[tuple]:
        """Generate 'Did X die before Y was born?' question"""
        # Pick two people with complete date info
        person1 = self.pick_random_person(
            filter_fn=lambda p: p['date_of_birth'] and p['date_of_death']
        )
        person2 = self.pick_random_person(
            filter_fn=lambda p: p['date_of_birth'] and p['canonical_name'] != person1
        )

        if not person1 or not person2:
            return None

        death1 = self.data_loader.parse_date(self.data_loader.people[person1]['date_of_death'])
        birth2 = self.data_loader.parse_date(self.data_loader.people[person2]['date_of_birth'])

        if not death1 or not birth2:
            return None

        # Determine answer
        answer = "Yes" if death1 < birth2 else "No"

        question = f"Did {person1} die before {person2} was born?"

        metadata = {
            'question_type': 'lifespan_overlap',
            'people': [person1, person2],
            'requires_temporal_logic': True
        }

        return (question, answer, 'medium', metadata)

    def generate_age_at_death_question(self) -> Optional[tuple]:
        """Generate 'At what age did X die?' question"""
        # Pick person with complete date info
        person = self.pick_random_person(
            filter_fn=lambda p: p.get('lifespan_years') is not None
        )

        if not person:
            return None

        age = self.data_loader.people[person]['lifespan_years']
        question = f"At what age did {person} die?"
        answer = f"{age} years old"

        metadata = {
            'question_type': 'age_at_death',
            'people': [person],
            'requires_arithmetic': True,
            'numeric_answer': age
        }

        return (question, answer, 'easy', metadata)

    def generate_age_at_event_question(self) -> Optional[tuple]:
        """Generate 'How old was X when Y was born?' question"""
        # Pick two people where we can compare dates
        # CRITICAL: Person1 must be ALIVE when person2 was born
        person1 = self.pick_random_person(
            filter_fn=lambda p: p['date_of_birth'] and p['date_of_death']
        )
        person2 = self.pick_random_person(
            filter_fn=lambda p: p['date_of_birth'] and p['canonical_name'] != person1
        )

        if not person1 or not person2:
            return None

        birth1 = self.data_loader.parse_date(self.data_loader.people[person1]['date_of_birth'])
        birth2 = self.data_loader.parse_date(self.data_loader.people[person2]['date_of_birth'])
        death1 = self.data_loader.parse_date(self.data_loader.people[person1]['date_of_death'])

        if not birth1 or not birth2 or not death1:
            return None

        # VERIFY: Person1 must be born before person2 AND alive when person2 was born
        if not (birth1 < birth2 < death1):
            return None  # Person1 either not born yet or already dead

        # Calculate age difference
        age_diff = (birth2 - birth1).days // 365

        question = f"How old was {person1} when {person2} was born?"
        answer = f"{age_diff} years old"

        metadata = {
            'question_type': 'age_at_event',
            'people': [person1, person2],
            'requires_arithmetic': True,
            'numeric_answer': age_diff
        }

        return (question, answer, 'medium', metadata)

    def generate_reign_during_birth_question(self) -> Optional[tuple]:
        """Generate 'Who reigned when X was born?' question"""
        # Pick a person with birth date
        person = self.pick_random_person(
            filter_fn=lambda p: p['date_of_birth']
        )

        if not person:
            return None

        birth_date = self.data_loader.parse_date(self.data_loader.people[person]['date_of_birth'])
        if not birth_date:
            return None

        # Find monarchs who reigned during this date
        reigning_monarchs = []
        for monarch_name, monarch_data in self.data_loader.people.items():
            if monarch_data['reign_start'] and monarch_data['reign_end']:
                reign_start = self.data_loader.parse_date(monarch_data['reign_start'])
                reign_end = self.data_loader.parse_date(monarch_data['reign_end'])

                if reign_start and reign_end:
                    if reign_start <= birth_date <= reign_end:
                        reigning_monarchs.append(monarch_name)

        if not reigning_monarchs:
            return None

        # Pick one monarch (there should usually be only one)
        monarch = reigning_monarchs[0]
        question = f"Who was reigning when {person} was born?"
        answer = monarch

        metadata = {
            'question_type': 'reign_during_birth',
            'people': [person, monarch],
            'requires_temporal_logic': True
        }

        return (question, answer, 'hard', metadata)


if __name__ == '__main__':
    # Test the task generator
    from .data_loader import RoyalFamilyDataLoader

    print("Testing Task 2: Temporal Reasoning")
    print("="*80)

    loader = RoyalFamilyDataLoader()
    task = TemporalReasoningTask(loader)

    # Generate sample questions
    print("\nGenerating 10 sample questions...")
    questions = task.generate_questions(count=10, difficulty='mixed')

    for i, q in enumerate(questions, 1):
        print(f"\n{i}. [{q['difficulty'].upper()}] - {q['metadata']['question_type']}")
        print(f"   Q: {q['question']}")
        print(f"   A: {q['answer']}")

    print(f"\n✓ Successfully generated {len(questions)} questions")
