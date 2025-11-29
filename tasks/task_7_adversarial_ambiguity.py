"""
Task 7: Adversarial Ambiguity
Tests LLM ability to disambiguate people with similar names across dynasties.
"""

from .task_base import TaskGenerator
from typing import List, Dict, Optional
import random
from collections import defaultdict


class AdversarialAmbiguityTask(TaskGenerator):
    """
    Generate questions with intentionally confusing similar names or dates.
    Examples:
    - "Edward III had a son Edward. Which Edward died first?"
    - "Margaret Tudor married a Henry. Margaret of Anjou also married a Henry. Did these Henrys ever meet?"

    Difficulty levels:
    - Easy: 2 people with same first name
    - Medium: 3 people with same first name
    - Hard: Complex scenarios with multiple overlapping names
    """

    def __init__(self, data_loader):
        super().__init__(
            data_loader=data_loader,
            task_name="Adversarial Ambiguity",
            task_type="adversarial_ambiguity"
        )

    def generate_questions(self, count: int, difficulty: str = 'mixed') -> List[Dict]:
        """Generate adversarial ambiguity questions"""
        questions = []
        seen_questions = set()

        # Question types
        question_types = [
            'same_name_death_order',
            'same_name_birth_order',
            'same_name_relationship',
            'common_spouse_name',
            'similar_dates_confusion'
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
        max_attempts = count * 150

        while len(questions) < count and attempts < max_attempts:
            attempts += 1

            # Get difficulty for this question
            current_diff = difficulty_schedule[len(questions)] if len(questions) < len(difficulty_schedule) else difficulty

            q_type = random.choice(question_types)

            result = None
            if q_type == 'same_name_death_order':
                result = self.generate_same_name_death_order_question(current_diff)
            elif q_type == 'same_name_birth_order':
                result = self.generate_same_name_birth_order_question(current_diff)
            elif q_type == 'same_name_relationship':
                result = self.generate_same_name_relationship_question(current_diff)
            elif q_type == 'common_spouse_name':
                result = self.generate_common_spouse_name_question(current_diff)
            elif q_type == 'similar_dates_confusion':
                result = self.generate_similar_dates_confusion_question(current_diff)

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

    def find_people_with_common_first_name(self, min_count=2):
        """Find groups of people sharing the same first name"""
        first_names = defaultdict(list)

        for person_name in self.data_loader.people.keys():
            # Extract first name (before any comma or parenthesis)
            first_name = person_name.split('(')[0].split(',')[0].strip().split()[0]
            first_names[first_name].append(person_name)

        # Return only names with at least min_count people
        return {name: people for name, people in first_names.items() if len(people) >= min_count}

    def generate_same_name_death_order_question(self, difficulty='medium') -> Optional[tuple]:
        """
        Generate: "Edward III had a son Edward who had a son Edward. Which Edward died first?"
        """
        common_names = self.find_people_with_common_first_name(min_count=3)

        if not common_names:
            return None

        # Try to find a 3-generation chain with the same first name
        for first_name, people in common_names.items():
            for person in people:
                children = self.data_loader.get_children(person)

                for child in children:
                    if first_name in child:
                        grandchildren = self.data_loader.get_children(child)

                        for grandchild in grandchildren:
                            if first_name in grandchild:
                                # Found 3 generations with same first name!
                                # Get death dates
                                person_death = self.data_loader.parse_date(
                                    self.data_loader.people[person]['date_of_death']
                                )
                                child_death = self.data_loader.parse_date(
                                    self.data_loader.people[child]['date_of_death']
                                )
                                grandchild_death = self.data_loader.parse_date(
                                    self.data_loader.people[grandchild]['date_of_death']
                                )

                                if person_death and child_death and grandchild_death:
                                    # Determine who died first
                                    deaths = [
                                        (person, person_death.year, "grandfather"),
                                        (child, child_death.year, "father"),
                                        (grandchild, grandchild_death.year, "son")
                                    ]
                                    deaths.sort(key=lambda x: x[1])
                                    first_to_die = deaths[0][0]

                                    question = (f"{person} had a son {child}, who had a son {grandchild}. "
                                              f"Which {first_name} died first: the grandfather, father, or son?")

                                    # Answer with role
                                    answer = deaths[0][2]

                                    metadata = {
                                        'question_type': 'same_name_death_order',
                                        'first_name': first_name,
                                        'people': [person, child, grandchild],
                                        'correct_answer': first_to_die,
                                        'requires_disambiguation': True
                                    }

                                    return (question, answer, 'hard', metadata)

        return None

    def generate_same_name_birth_order_question(self, difficulty='medium') -> Optional[tuple]:
        """Generate: "There were 3 people named Henry. Which Henry was born first?"""""
        common_names = self.find_people_with_common_first_name(min_count=3)

        if not common_names:
            return None

        # Pick a common first name
        first_name = random.choice(list(common_names.keys()))
        people = common_names[first_name]

        # Filter people with birth dates
        people_with_birth = []
        for person in people:
            birth = self.data_loader.parse_date(self.data_loader.people[person]['date_of_birth'])
            if birth:
                people_with_birth.append((person, birth.year))

        if len(people_with_birth) < 3:
            return None

        # Pick 3 random people
        selected = random.sample(people_with_birth, 3)
        selected.sort(key=lambda x: x[1])  # Sort by birth year

        earliest_person = selected[0][0]
        people_names = [p[0] for p in selected]

        question = f"Three people named {first_name}: {', '.join(people_names)}. Which {first_name} was born first?"
        answer = earliest_person

        metadata = {
            'question_type': 'same_name_birth_order',
            'first_name': first_name,
            'people': people_names,
            'correct_answer': earliest_person,
            'requires_disambiguation': True
        }

        return (question, answer, 'medium', metadata)

    def generate_same_name_relationship_question(self, difficulty='medium') -> Optional[tuple]:
        """Generate: "Edward I had a grandson Edward. Is Edward II the grandson?"""""
        common_names = self.find_people_with_common_first_name(min_count=2)

        if not common_names:
            return None

        # Find a grandparent-grandchild pair with same first name
        for first_name, people in common_names.items():
            for person in people:
                # Get grandchildren
                children = self.data_loader.get_children(person)
                for child in children:
                    grandchildren = self.data_loader.get_children(child)
                    for grandchild in grandchildren:
                        if first_name in grandchild:
                            # Found a match!
                            # Pick another person with same first name as distractor
                            other_people = [p for p in people if p != person and p != grandchild]
                            if other_people:
                                distractor = random.choice(other_people)

                                # Ask if distractor is the grandchild
                                question = f"{person} had a grandchild named {first_name}. Is {distractor} that grandchild?"
                                answer = "No" if distractor != grandchild else "Yes"

                                metadata = {
                                    'question_type': 'same_name_relationship',
                                    'first_name': first_name,
                                    'grandparent': person,
                                    'actual_grandchild': grandchild,
                                    'asked_about': distractor,
                                    'requires_disambiguation': True
                                }

                                return (question, answer, 'hard', metadata)

        return None

    def generate_common_spouse_name_question(self, difficulty='hard') -> Optional[tuple]:
        """
        Generate: "Margaret Tudor married a Henry. Margaret of Anjou also married a Henry.
        Did these two Henrys ever meet?"
        """
        # Find people with spouses that share a common first name
        spouse_first_names = defaultdict(list)  # first_name -> [(person, spouse)]

        for person_name, person_data in self.data_loader.people.items():
            if person_data.get('spouse'):
                spouse_name = person_data['spouse']
                spouse = self.data_loader.find_person(spouse_name)
                if spouse:
                    spouse_first_name = spouse.split('(')[0].split(',')[0].strip().split()[0]
                    spouse_first_names[spouse_first_name].append((person_name, spouse))

        # Find a first name with at least 2 different spouses
        for first_name, pairs in spouse_first_names.items():
            if len(pairs) >= 2:
                # Pick 2 pairs
                pair1, pair2 = random.sample(pairs, 2)
                person1, spouse1 = pair1
                person2, spouse2 = pair2

                if spouse1 == spouse2:
                    continue  # Same spouse, not interesting

                # Check if the two spouses could have met (lifespan overlap)
                spouse1_data = self.data_loader.people.get(spouse1)
                spouse2_data = self.data_loader.people.get(spouse2)

                if not spouse1_data or not spouse2_data:
                    continue

                birth1 = self.data_loader.parse_date(spouse1_data['date_of_birth'])
                death1 = self.data_loader.parse_date(spouse1_data['date_of_death'])
                birth2 = self.data_loader.parse_date(spouse2_data['date_of_birth'])
                death2 = self.data_loader.parse_date(spouse2_data['date_of_death'])

                if not (birth1 and birth2):
                    continue

                # Check lifespan overlap
                could_have_met = False
                if death1 and death2:
                    could_have_met = (birth1.year <= death2.year and birth2.year <= death1.year)
                elif death1:
                    could_have_met = (birth2.year <= death1.year)
                elif death2:
                    could_have_met = (birth1.year <= death2.year)
                else:
                    could_have_met = True

                question = (f"{person1} married a {first_name} ({spouse1}). "
                          f"{person2} also married a {first_name} ({spouse2}). "
                          f"Did these two people named {first_name} ever meet?")

                answer = "Possibly yes (lifespans overlapped)" if could_have_met else "No (lifespans did not overlap)"

                metadata = {
                    'question_type': 'common_spouse_name',
                    'first_name': first_name,
                    'people': [person1, person2],
                    'spouses': [spouse1, spouse2],
                    'could_have_met': could_have_met,
                    'requires_disambiguation': True,
                    'requires_temporal_logic': True
                }

                return (question, answer, 'hard', metadata)

        return None

    def generate_similar_dates_confusion_question(self, difficulty='hard') -> Optional[tuple]:
        """Generate questions with similar dates that cause confusion"""
        # Find people born in adjacent years
        people_by_birth_year = defaultdict(list)

        for person_name, person_data in self.data_loader.people.items():
            birth = self.data_loader.parse_date(person_data['date_of_birth'])
            if birth:
                people_by_birth_year[birth.year].append(person_name)

        # Find adjacent years with people
        years = sorted(people_by_birth_year.keys())

        for i in range(len(years) - 1):
            year1, year2 = years[i], years[i + 1]

            if abs(year1 - year2) <= 2 and people_by_birth_year[year1] and people_by_birth_year[year2]:
                person1 = random.choice(people_by_birth_year[year1])
                person2 = random.choice(people_by_birth_year[year2])

                question = f"{person1} was born in {year1}. {person2} was born in {year2}. Who was born first?"
                answer = person1 if year1 < year2 else person2

                metadata = {
                    'question_type': 'similar_dates_confusion',
                    'people': [person1, person2],
                    'years': [year1, year2],
                    'correct_answer': answer,
                    'year_difference': abs(year1 - year2)
                }

                return (question, answer, 'easy', metadata)

        return None


if __name__ == '__main__':
    # Test the task generator
    from .data_loader import RoyalFamilyDataLoader

    print("Testing Task 9: Adversarial Ambiguity")
    print("="*80)

    loader = RoyalFamilyDataLoader()
    task = AdversarialAmbiguityTask(loader)

    # Generate sample questions
    print("\nGenerating 10 sample questions (mixed difficulty)...")
    questions = task.generate_questions(count=10, difficulty='mixed')

    for i, q in enumerate(questions, 1):
        print(f"\n{i}. [{q['difficulty'].upper()}] - {q['metadata']['question_type']}")
        print(f"   Q: {q['question'][:120]}...")
        print(f"   A: {q['answer']}")

    print(f"\n✓ Successfully generated {len(questions)} questions")
