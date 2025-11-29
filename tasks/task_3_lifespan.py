"""
Task 3: Comparative Lifespan Reasoning
Tests numerical comparison and lifespan arithmetic.
"""

from .task_base import TaskGenerator
from typing import List, Dict, Optional
import random


class ComparativeLifespanTask(TaskGenerator):
    """
    Generate questions comparing lifespans and ages.
    Examples:
    - "Who lived longer, X or Y?"
    - "Who was older when they became king, X or Y?"
    - "Who lived to an older age, X or Y, and by how many years?"

    Difficulty levels:
    - Easy: Simple 2-person comparisons
    - Medium: 3-person comparisons, age at event
    - Hard: 4-5 person ranking, complex conditions
    """

    def __init__(self, data_loader):
        super().__init__(
            data_loader=data_loader,
            task_name="Comparative Lifespan Reasoning",
            task_type="comparative_lifespan"
        )

    def generate_questions(self, count: int, difficulty: str = 'mixed') -> List[Dict]:
        """Generate comparative lifespan questions"""
        questions = []
        seen_questions = set()

        # Question types
        question_types = [
            'who_lived_longer',
            'age_difference',
            'older_at_coronation',
            'older_when_born',
            'who_was_older_when_died'
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
            if q_type == 'who_lived_longer':
                result = self.generate_who_lived_longer_question(current_diff)
            elif q_type == 'age_difference':
                result = self.generate_age_difference_question()
            elif q_type == 'older_at_coronation':
                result = self.generate_older_at_coronation_question()
            elif q_type == 'older_when_born':
                result = self.generate_older_when_born_question()
            elif q_type == 'who_was_older_when_died':
                result = self.generate_who_was_older_when_died_question(current_diff)

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

    def generate_who_lived_longer_question(self, difficulty='easy') -> Optional[tuple]:
        """Generate 'Who lived longer?' question (2-5 people depending on difficulty)"""
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

        # Pick people with known lifespans
        people_with_lifespans = [
            (p['canonical_name'], p['lifespan_years'])
            for p in self.data_loader.people.values()
            if p.get('lifespan_years') is not None
        ]

        if len(people_with_lifespans) < num_people:
            return None

        # Pick random people and sort by lifespan
        selected = random.sample(people_with_lifespans, num_people)
        selected.sort(key=lambda x: x[1], reverse=True)  # Sort by lifespan (longest first)

        people_names = [name for name, _ in selected]
        lifespans = [lifespan for _, lifespan in selected]

        if num_people == 2:
            question = f"Who lived longer, {people_names[0]} or {people_names[1]}?"
            answer = f"{people_names[0]} (lived {lifespans[0]} years vs {lifespans[1]} years)"
        elif num_people == 3:
            question = f"Who lived longest among {people_names[0]}, {people_names[1]}, and {people_names[2]}?"
            answer = f"{people_names[0]} (lived {lifespans[0]} years)"
        else:  # 4-5 people
            people_str = ', '.join(people_names[:-1]) + f', and {people_names[-1]}'
            question = f"Rank the following people by lifespan (longest to shortest): {people_str}"
            answer = ', '.join(people_names)  # Already sorted

        metadata = {
            'question_type': 'who_lived_longer',
            'people': people_names,
            'lifespans': lifespans,
            'num_people': num_people,
            'requires_comparison': True,
            'requires_ranking': num_people > 3
        }

        return (question, answer, diff_level, metadata)

    def generate_age_difference_question(self) -> Optional[tuple]:
        """Generate 'Who lived to an older age, and by how many years?' question"""
        # Pick two people with known lifespans
        person1 = self.pick_random_person(
            filter_fn=lambda p: p.get('lifespan_years') is not None
        )
        person2 = self.pick_random_person(
            filter_fn=lambda p: p.get('lifespan_years') is not None and p['canonical_name'] != person1
        )

        if not person1 or not person2:
            return None

        lifespan1 = self.data_loader.people[person1]['lifespan_years']
        lifespan2 = self.data_loader.people[person2]['lifespan_years']

        difference = abs(lifespan1 - lifespan2)

        if difference == 0:
            return None

        # Unambiguous phrasing: compare lifespans, not temporal "outliving"
        question = f"Who lived to an older age, {person1} or {person2}, and by how many years?"

        if lifespan1 > lifespan2:
            answer = f"{person1} lived {difference} years longer (lived {lifespan1} years vs {lifespan2} years)"
        else:
            answer = f"{person2} lived {difference} years longer (lived {lifespan2} years vs {lifespan1} years)"

        metadata = {
            'question_type': 'age_difference',
            'people': [person1, person2],
            'lifespans': [lifespan1, lifespan2],
            'difference': difference,
            'requires_arithmetic': True
        }

        return (question, answer, 'medium', metadata)

    def generate_older_at_coronation_question(self) -> Optional[tuple]:
        """Generate 'Who was older when crowned, X or Y?' question"""
        # Pick two monarchs with coronation age data
        person1 = self.pick_random_person(
            filter_fn=lambda p: p.get('age_at_coronation') is not None
        )
        person2 = self.pick_random_person(
            filter_fn=lambda p: p.get('age_at_coronation') is not None and p['canonical_name'] != person1
        )

        if not person1 or not person2:
            return None

        age1 = self.data_loader.people[person1]['age_at_coronation']
        age2 = self.data_loader.people[person2]['age_at_coronation']

        if age1 == age2:
            return None

        if age1 > age2:
            answer = f"{person1} (was {age1} years old vs {age2} years old)"
        else:
            answer = f"{person2} (was {age2} years old vs {age1} years old)"

        question = f"Who was older when they were crowned, {person1} or {person2}?"

        metadata = {
            'question_type': 'older_at_coronation',
            'people': [person1, person2],
            'ages_at_coronation': [age1, age2],
            'requires_comparison': True,
            'requires_context': True  # Need to know coronation ages
        }

        return (question, answer, 'medium', metadata)

    def generate_older_when_born_question(self) -> Optional[tuple]:
        """Generate 'Who was older when X was born, Y or Z?' question"""
        # Pick three people with birth dates
        person_x = self.pick_random_person(
            filter_fn=lambda p: p['date_of_birth']
        )
        person_y = self.pick_random_person(
            filter_fn=lambda p: p['date_of_birth'] and p['canonical_name'] != person_x
        )
        person_z = self.pick_random_person(
            filter_fn=lambda p: p['date_of_birth'] and p['canonical_name'] not in [person_x, person_y]
        )

        if not all([person_x, person_y, person_z]):
            return None

        birth_x = self.data_loader.parse_date(self.data_loader.people[person_x]['date_of_birth'])
        birth_y = self.data_loader.parse_date(self.data_loader.people[person_y]['date_of_birth'])
        birth_z = self.data_loader.parse_date(self.data_loader.people[person_z]['date_of_birth'])

        if not all([birth_x, birth_y, birth_z]):
            return None

        # Calculate ages when person_x was born
        age_y_when_x_born = (birth_x - birth_y).days // 365 if birth_y < birth_x else None
        age_z_when_x_born = (birth_x - birth_z).days // 365 if birth_z < birth_x else None

        if age_y_when_x_born is None or age_z_when_x_born is None:
            return None

        if age_y_when_x_born <= 0 or age_z_when_x_born <= 0:
            return None

        if age_y_when_x_born == age_z_when_x_born:
            return None

        if age_y_when_x_born > age_z_when_x_born:
            answer = f"{person_y} (was {age_y_when_x_born} years old vs {age_z_when_x_born} years old)"
        else:
            answer = f"{person_z} (was {age_z_when_x_born} years old vs {age_y_when_x_born} years old)"

        question = f"Who was older when {person_x} was born, {person_y} or {person_z}?"

        metadata = {
            'question_type': 'older_when_born',
            'people': [person_x, person_y, person_z],
            'ages': [age_y_when_x_born, age_z_when_x_born],
            'requires_arithmetic': True,
            'requires_temporal_logic': True
        }

        return (question, answer, 'hard', metadata)

    def generate_who_was_older_when_died_question(self, difficulty='easy') -> Optional[tuple]:
        """Generate 'Who was older when they died?' question (2-5 people depending on difficulty)"""
        # Determine number of people
        if difficulty == 'hard':
            num_people = random.choice([4, 5])
            diff_level = 'hard'
        elif difficulty == 'medium':
            num_people = 3
            diff_level = 'medium'
        else:
            num_people = 2
            diff_level = 'easy'

        # Pick people with known lifespans
        people_with_lifespans = [
            (p['canonical_name'], p['lifespan_years'])
            for p in self.data_loader.people.values()
            if p.get('lifespan_years') is not None
        ]

        if len(people_with_lifespans) < num_people:
            return None

        # Pick random people and sort by age at death
        selected = random.sample(people_with_lifespans, num_people)
        selected.sort(key=lambda x: x[1], reverse=True)  # Sort by age (oldest first)

        people_names = [name for name, _ in selected]
        ages = [age for _, age in selected]

        if num_people == 2:
            question = f"Who was older when they died, {people_names[0]} or {people_names[1]}?"
            answer = f"{people_names[0]} (died at {ages[0]} vs {ages[1]})"
        elif num_people == 3:
            question = f"Who was oldest when they died among {people_names[0]}, {people_names[1]}, and {people_names[2]}?"
            answer = f"{people_names[0]} (died at age {ages[0]})"
        else:  # 4-5 people
            people_str = ', '.join(people_names[:-1]) + f', and {people_names[-1]}'
            question = f"Rank the following by age at death (oldest to youngest): {people_str}"
            answer = ', '.join(people_names)

        metadata = {
            'question_type': 'who_was_older_when_died',
            'people': people_names,
            'ages_at_death': ages,
            'num_people': num_people,
            'requires_comparison': True,
            'requires_ranking': num_people > 3
        }

        return (question, answer, diff_level, metadata)


if __name__ == '__main__':
    # Test the task generator
    from .data_loader import RoyalFamilyDataLoader

    print("Testing Task 5: Comparative Lifespan Reasoning")
    print("="*80)

    loader = RoyalFamilyDataLoader()
    task = ComparativeLifespanTask(loader)

    # Generate sample questions
    print("\nGenerating 10 sample questions...")
    questions = task.generate_questions(count=10, difficulty='mixed')

    for i, q in enumerate(questions, 1):
        print(f"\n{i}. [{q['difficulty'].upper()}] - {q['metadata']['question_type']}")
        print(f"   Q: {q['question']}")
        print(f"   A: {q['answer']}")

    print(f"\n✓ Successfully generated {len(questions)} questions")
