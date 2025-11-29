"""
Task 4: Negative Reasoning
Tests ability to handle negation and false premises.
"""

from .task_base import TaskGenerator
from typing import List, Dict, Optional
import random


class NegativeReasoningTask(TaskGenerator):
    """
    Generate questions requiring negative reasoning.
    Examples:
    - "Who among [A, B, C] is NOT a child of X?"
    - "True or False: X is the grandparent of Y"
    - "Which of these people did NOT live in the 1800s?"

    Difficulty levels:
    - Easy: Simple negation with 2-3 options
    - Medium: Multiple choice with 4-5 options
    - Hard: Complex conditions, multiple negations
    """

    def __init__(self, data_loader):
        super().__init__(
            data_loader=data_loader,
            task_name="Negative Reasoning",
            task_type="negative"
        )

    def generate_questions(self, count: int, difficulty: str = 'mixed') -> List[Dict]:
        """Generate negative reasoning questions"""
        questions = []
        seen_questions = set()

        # Question types by difficulty
        easy_types = ['not_child_of', 'true_false_relationship', 'not_parent_of']
        medium_types = ['not_child_of', 'not_lived_in_period', 'not_sibling_of', 'true_false_relationship']
        hard_types = ['double_negative_monarch', 'double_negative_parent', 'implicit_negation_meeting',
                      'false_premise', 'nested_negation_triple', 'multi_condition_not']

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

            # Pick question type based on difficulty
            if current_diff == 'hard':
                question_types = hard_types
            elif current_diff == 'medium':
                question_types = medium_types
            else:
                question_types = easy_types

            q_type = random.choice(question_types)

            result = None
            if q_type == 'not_child_of':
                result = self.generate_not_child_of_question(current_diff)
            elif q_type == 'true_false_relationship':
                result = self.generate_true_false_question()
            elif q_type == 'not_lived_in_period':
                result = self.generate_not_lived_in_period_question(current_diff)
            elif q_type == 'not_parent_of':
                result = self.generate_not_parent_of_question()
            elif q_type == 'not_sibling_of':
                result = self.generate_not_sibling_of_question(current_diff)
            elif q_type == 'false_premise':
                result = self.generate_false_premise_question()
            elif q_type == 'double_negative_monarch':
                result = self.generate_double_negative_monarch_question()
            elif q_type == 'double_negative_parent':
                result = self.generate_double_negative_parent_question()
            elif q_type == 'implicit_negation_meeting':
                result = self.generate_implicit_negation_question()
            elif q_type == 'nested_negation_triple':
                result = self.generate_nested_negation_triple_question()
            elif q_type == 'multi_condition_not':
                result = self.generate_multi_condition_not_question()

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

    def generate_not_child_of_question(self, difficulty='easy') -> Optional[tuple]:
        """Generate 'Who among [A, B, C] is NOT a child of X?' question"""
        # Pick a person with at least 2 children
        parent = self.pick_random_person(
            filter_fn=lambda p: len(self.data_loader.get_children(p['canonical_name'])) >= 2
        )

        if not parent:
            return None

        children = self.data_loader.get_children(parent)
        parent_data = self.data_loader.people.get(parent)

        # More options and better distractors for medium/hard
        if difficulty == 'medium':
            num_children = min(3, len(children))
            num_non_children = 2
        else:  # easy
            num_children = min(2, len(children))
            num_non_children = 1

        # Pick actual children
        actual_children = random.sample(children, num_children)

        # For medium, pick distractors from same era/house (not random people from different centuries)
        if difficulty == 'medium' and parent_data:
            # Get siblings of the parent (would be aunts/uncles of the children - plausible confusion)
            parent_siblings = self.data_loader.get_siblings(parent)
            if parent_siblings and len(parent_siblings) >= num_non_children:
                non_children = random.sample(list(parent_siblings), num_non_children)
            else:
                # Fallback to random but same house
                house = parent_data.get('house', '')
                non_children_candidates = [
                    p['canonical_name'] for p in self.data_loader.people.values()
                    if p['canonical_name'] not in children
                    and p['canonical_name'] != parent
                    and p.get('house') == house
                ]
                if len(non_children_candidates) >= num_non_children:
                    non_children = random.sample(non_children_candidates, num_non_children)
                else:
                    return None
        else:
            # Easy: just pick random non-child
            non_child = self.pick_random_person(
                filter_fn=lambda p: p['canonical_name'] not in children and p['canonical_name'] != parent
            )
            if not non_child:
                return None
            non_children = [non_child]

        # Create the options list
        options = actual_children + non_children
        random.shuffle(options)

        question = f"Who among {', '.join(options)} is NOT a child of {parent}?"
        answer = non_children[0] if len(non_children) == 1 else ', '.join(sorted(non_children))

        metadata = {
            'question_type': 'not_child_of',
            'parent': parent,
            'options': options,
            'correct_answer': answer,
            'requires_negation': True
        }

        diff_level = 'medium' if difficulty == 'medium' else 'easy'
        return (question, answer, diff_level, metadata)

    def generate_true_false_question(self) -> Optional[tuple]:
        """Generate 'True or False: X is the [relationship] of Y' question"""
        # Pick a relationship to test
        relationships = ['grandparent', 'parent', 'sibling', 'child']
        relationship = random.choice(relationships)

        person1 = self.pick_random_person()
        person2 = self.pick_random_person(
            filter_fn=lambda p: p['canonical_name'] != person1
        )

        if not person1 or not person2:
            return None

        # Check if the relationship is actually true
        is_true = False

        if relationship == 'parent':
            mother, father = self.data_loader.get_parents(person2)
            is_true = person1 in [mother, father]

        elif relationship == 'child':
            children = self.data_loader.get_children(person1)
            is_true = person2 in children

        elif relationship == 'sibling':
            siblings = self.data_loader.get_siblings(person1)
            is_true = person2 in siblings

        elif relationship == 'grandparent':
            # Check if person1 is grandparent of person2
            mother, father = self.data_loader.get_parents(person2)
            for parent in [mother, father]:
                if parent:
                    gm, gf = self.data_loader.get_parents(parent)
                    if person1 in [gm, gf]:
                        is_true = True
                        break

        answer = "True" if is_true else "False"

        question = f"True or False: {person1} is the {relationship} of {person2}."

        metadata = {
            'question_type': 'true_false_relationship',
            'people': [person1, person2],
            'relationship': relationship,
            'is_true': is_true,
            'requires_verification': True
        }

        difficulty = 'easy' if relationship in ['parent', 'child'] else 'medium'

        return (question, answer, difficulty, metadata)

    def generate_not_lived_in_period_question(self, difficulty='medium') -> Optional[tuple]:
        """Generate 'Which of these people did NOT live in the [century]?' question"""
        # Pick a century
        centuries = [
            (900, 999, "10th century"),
            (1000, 1099, "11th century"),
            (1100, 1199, "12th century"),
            (1200, 1299, "13th century"),
            (1300, 1399, "14th century"),
            (1400, 1499, "15th century"),
            (1500, 1599, "16th century"),
            (1600, 1699, "17th century"),
            (1700, 1799, "18th century"),
            (1800, 1899, "19th century"),
            (1900, 1999, "20th century"),
        ]

        start_year, end_year, century_name = random.choice(centuries)

        # Find people who lived in this period
        people_in_period = []
        people_not_in_period = []

        for person_name, person_data in self.data_loader.people.items():
            birth = self.data_loader.parse_date(person_data['date_of_birth'])
            death = self.data_loader.parse_date(person_data['date_of_death'])

            if birth:
                birth_year = birth.year
                # Person lived in period if born during or before end, and (no death date or died after start)
                if birth_year <= end_year:
                    if not death or death.year >= start_year:
                        people_in_period.append(person_name)
                        continue

            people_not_in_period.append(person_name)

        if len(people_in_period) < 2 or len(people_not_in_period) < 1:
            return None

        # Pick 2-3 people from the period and 1 not from the period
        in_period = random.sample(people_in_period, min(2, len(people_in_period)))
        not_in_period = random.choice(people_not_in_period)

        options = in_period + [not_in_period]
        random.shuffle(options)

        question = f"Which of these people did NOT live during the {century_name}: {', '.join(options)}?"
        answer = not_in_period

        metadata = {
            'question_type': 'not_lived_in_period',
            'period': century_name,
            'options': options,
            'correct_answer': not_in_period,
            'requires_negation': True,
            'requires_temporal_logic': True
        }

        return (question, answer, 'medium', metadata)

    def generate_not_parent_of_question(self) -> Optional[tuple]:
        """Generate 'Who is NOT a parent of X?' question"""
        # Pick a person with at least one parent
        person = self.pick_random_person(
            filter_fn=lambda p: self.data_loader.get_parents(p['canonical_name'])[0] or
                               self.data_loader.get_parents(p['canonical_name'])[1]
        )

        if not person:
            return None

        mother, father = self.data_loader.get_parents(person)
        parents = [p for p in [mother, father] if p]

        if not parents:
            return None

        # Pick a person who is NOT a parent
        non_parent = self.pick_random_person(
            filter_fn=lambda p: p['canonical_name'] not in parents and p['canonical_name'] != person
        )

        if not non_parent:
            return None

        # Create options
        options = parents + [non_parent]
        random.shuffle(options)

        question = f"Who among {', '.join(options)} is NOT a parent of {person}?"
        answer = non_parent

        metadata = {
            'question_type': 'not_parent_of',
            'person': person,
            'options': options,
            'correct_answer': non_parent,
            'requires_negation': True
        }

        return (question, answer, 'easy', metadata)

    def generate_not_sibling_of_question(self, difficulty='medium') -> Optional[tuple]:
        """Generate 'Who is NOT a sibling of X?' question"""
        # Pick a person with siblings
        person = self.pick_random_person(
            filter_fn=lambda p: len(self.data_loader.get_siblings(p['canonical_name'])) >= 1
        )

        if not person:
            return None

        siblings = list(self.data_loader.get_siblings(person))

        # Pick 1-2 actual siblings
        actual_siblings = random.sample(siblings, min(2, len(siblings)))

        # Pick a non-sibling
        non_sibling = self.pick_random_person(
            filter_fn=lambda p: p['canonical_name'] not in siblings and p['canonical_name'] != person
        )

        if not non_sibling:
            return None

        options = actual_siblings + [non_sibling]
        random.shuffle(options)

        question = f"Who among {', '.join(options)} is NOT a sibling of {person}?"
        answer = non_sibling

        metadata = {
            'question_type': 'not_sibling_of',
            'person': person,
            'options': options,
            'correct_answer': non_sibling,
            'requires_negation': True,
            'requires_inference': True  # Siblings not directly encoded
        }

        return (question, answer, 'medium', metadata)

    def generate_false_premise_question(self) -> Optional[tuple]:
        """Generate a question with a false premise that should be rejected"""
        # Pick two unrelated people
        person1 = self.pick_random_person()
        person2 = self.pick_random_person(
            filter_fn=lambda p: p['canonical_name'] != person1
        )

        if not person1 or not person2:
            return None

        # Check they're not actually related
        gen_dist = self.data_loader.compute_generation_distance(person1, person2)
        if gen_dist is not None:
            return None  # They are related, skip

        # Create false premise
        false_premises = [
            f"How many children did {person1} and {person2} have together?",
            f"When did {person1} marry {person2}?",
            f"Which child of {person1} and {person2} became king?"
        ]

        question = random.choice(false_premises)
        answer = "False premise: there is no evidence that these two people were married or had children together."

        metadata = {
            'question_type': 'false_premise',
            'people': [person1, person2],
            'requires_premise_rejection': True
        }

        return (question, answer, 'hard', metadata)

    def generate_double_negative_monarch_question(self) -> Optional[tuple]:
        """Generate double negative: 'Who is NOT a person who did NOT become monarch?'"""
        # Find people who became monarchs
        monarchs = [
            p['canonical_name'] for p in self.data_loader.people.values()
            if p.get('reign_start')
        ]

        # Find people who did NOT become monarchs
        non_monarchs = [
            p['canonical_name'] for p in self.data_loader.people.values()
            if not p.get('reign_start')
        ]

        if len(monarchs) < 2 or len(non_monarchs) < 2:
            return None

        # Pick 2 monarchs and 2 non-monarchs from same era
        selected_monarchs = random.sample(monarchs, 2)
        selected_non_monarchs = random.sample(non_monarchs, 2)

        options = selected_monarchs + selected_non_monarchs
        random.shuffle(options)

        # Double negative: "NOT a person who did NOT become monarch" = "became monarch"
        # So correct answers are the monarchs
        question = f"Who among {', '.join(options)} is NOT a person who did NOT become monarch?"
        answer = ', '.join(sorted(selected_monarchs))

        metadata = {
            'question_type': 'double_negative_monarch',
            'options': options,
            'correct_answer': answer,
            'requires_double_negation': True
        }

        return (question, answer, 'hard', metadata)

    def generate_double_negative_parent_question(self) -> Optional[tuple]:
        """Generate: 'Who did NOT have a parent who was NOT a monarch?'"""
        # This means: who had at least one parent who WAS a monarch?

        # Find people with at least one parent who was a monarch
        people_with_monarch_parent = []
        people_without_monarch_parent = []

        for person_name, person_data in self.data_loader.people.items():
            mother, father = self.data_loader.get_parents(person_name)

            has_monarch_parent = False
            if mother:
                mother_data = self.data_loader.people.get(mother)
                if mother_data and mother_data.get('reign_start'):
                    has_monarch_parent = True
            if father:
                father_data = self.data_loader.people.get(father)
                if father_data and father_data.get('reign_start'):
                    has_monarch_parent = True

            if has_monarch_parent:
                people_with_monarch_parent.append(person_name)
            else:
                people_without_monarch_parent.append(person_name)

        if len(people_with_monarch_parent) < 2 or len(people_without_monarch_parent) < 2:
            return None

        # Pick options
        with_monarch_parent = random.sample(people_with_monarch_parent, 2)
        without_monarch_parent = random.sample(people_without_monarch_parent, 2)

        options = with_monarch_parent + without_monarch_parent
        random.shuffle(options)

        # "did NOT have a parent who was NOT a monarch" = "had a parent who WAS a monarch"
        question = f"Who among {', '.join(options)} did NOT have a parent who was NOT a monarch?"
        answer = ', '.join(sorted(with_monarch_parent))

        metadata = {
            'question_type': 'double_negative_parent',
            'options': options,
            'correct_answer': answer,
            'requires_double_negation': True
        }

        return (question, answer, 'hard', metadata)

    def generate_implicit_negation_question(self) -> Optional[tuple]:
        """Generate implicit negation: 'Who never met [person]?'"""
        # Pick a historical person
        reference_person = self.pick_random_person(
            filter_fn=lambda p: self.data_loader.parse_date(p['date_of_birth']) and
                               self.data_loader.parse_date(p['date_of_death'])
        )

        if not reference_person:
            return None

        ref_data = self.data_loader.people.get(reference_person)
        ref_birth = self.data_loader.parse_date(ref_data['date_of_birth'])
        ref_death = self.data_loader.parse_date(ref_data['date_of_death'])

        if not ref_birth or not ref_death:
            return None

        # Find people who lived during same time (could have met)
        could_have_met = []
        could_not_have_met = []

        for person_name, person_data in self.data_loader.people.items():
            if person_name == reference_person:
                continue

            birth = self.data_loader.parse_date(person_data['date_of_birth'])
            death = self.data_loader.parse_date(person_data['date_of_death'])

            if not birth:
                continue

            # Could have met if lifespans overlap
            if death:
                # Check if lifespans overlap
                if birth.year <= ref_death.year and death.year >= ref_birth.year:
                    could_have_met.append(person_name)
                else:
                    could_not_have_met.append(person_name)
            else:
                # No death date, assume could have met if born before reference died
                if birth.year <= ref_death.year:
                    could_have_met.append(person_name)

        if len(could_have_met) < 2 or len(could_not_have_met) < 2:
            return None

        # Pick options
        met = random.sample(could_have_met, 2)
        never_met = random.sample(could_not_have_met, 2)

        options = met + never_met
        random.shuffle(options)

        question = f"Who among {', '.join(options)} never met {reference_person} (based on their lifespans)?"
        answer = ', '.join(sorted(never_met))

        metadata = {
            'question_type': 'implicit_negation_meeting',
            'reference_person': reference_person,
            'options': options,
            'correct_answer': answer,
            'requires_temporal_logic': True,
            'requires_implicit_negation': True
        }

        return (question, answer, 'hard', metadata)

    def generate_nested_negation_triple_question(self) -> Optional[tuple]:
        """
        Generate triple negation question.
        Example: "Who did NOT have a child who did NOT marry someone who was NOT royalty?"
        Meaning: "Who had a child who married royalty?"
        """
        # Find people with children who married
        candidates_yes = []  # People who had a child who married royalty
        candidates_no = []   # People who did NOT have a child who married royalty

        for person_name in self.data_loader.people.keys():
            children = self.data_loader.get_children(person_name)

            if not children:
                candidates_no.append(person_name)
                continue

            # Check if any child married royalty (has a spouse with a royal house)
            has_child_married_royalty = False
            for child in children:
                child_data = self.data_loader.people.get(child)
                if child_data and child_data.get('spouse'):
                    spouse_name = child_data['spouse']
                    spouse = self.data_loader.find_person(spouse_name)
                    if spouse:
                        spouse_data = self.data_loader.people.get(spouse)
                        # Consider royalty if they have a reign_start or belong to a royal house
                        if spouse_data and (spouse_data.get('reign_start') or spouse_data.get('house')):
                            has_child_married_royalty = True
                            break

            if has_child_married_royalty:
                candidates_yes.append(person_name)
            else:
                candidates_no.append(person_name)

        if len(candidates_yes) < 2 or len(candidates_no) < 2:
            return None

        # Pick options
        yes_options = random.sample(candidates_yes, 2)
        no_options = random.sample(candidates_no, 2)

        options = yes_options + no_options
        random.shuffle(options)

        # Triple negative resolves to: "had a child who married royalty"
        question = f"Who among {', '.join(options)} did NOT have a child who did NOT marry someone who was NOT royalty?"
        answer = ', '.join(sorted(yes_options))

        metadata = {
            'question_type': 'nested_negation_triple',
            'options': options,
            'correct_answer': answer,
            'requires_triple_negation': True
        }

        return (question, answer, 'hard', metadata)

    def generate_multi_condition_not_question(self) -> Optional[tuple]:
        """
        Generate multi-condition NOT question.
        Example: "Who among [A,B,C,D] is NOT a child of X AND NOT born before 1400?"
        """
        # Pick a parent with children
        parent = self.pick_random_person(
            filter_fn=lambda p: len(self.data_loader.get_children(p['canonical_name'])) >= 2
        )

        if not parent:
            return None

        children = self.data_loader.get_children(parent)

        # Pick a year threshold
        year_threshold = random.choice([1200, 1300, 1400, 1500, 1600])

        # Find people born before threshold
        born_before = []
        born_after = []

        for person_name, person_data in self.data_loader.people.items():
            birth = self.data_loader.parse_date(person_data['date_of_birth'])
            if birth:
                if birth.year < year_threshold:
                    born_before.append(person_name)
                else:
                    born_after.append(person_name)

        # Find someone who is: (NOT a child of parent) AND (NOT born before threshold)
        # This means: not a child AND born on/after threshold
        correct_candidates = [p for p in born_after if p not in children and p != parent]

        if len(correct_candidates) < 1:
            return None

        # Pick the correct answer
        correct_answer = random.choice(correct_candidates)

        # Create distractors:
        # - Someone who IS a child (fails first condition)
        # - Someone who WAS born before threshold (fails second condition)
        # - Someone who fails both

        options = [correct_answer]

        if children:
            # Add a child of parent (fails NOT child condition)
            options.append(random.choice(children))

        # Add someone born before threshold (fails NOT born before condition)
        if born_before:
            non_child_born_before = [p for p in born_before if p not in children and p != parent]
            if non_child_born_before:
                options.append(random.choice(non_child_born_before))

        # Fill remaining slots with other people
        while len(options) < 4:
            filler = self.pick_random_person(
                filter_fn=lambda p: p['canonical_name'] not in options and p['canonical_name'] != parent
            )
            if filler:
                options.append(filler)
            else:
                break

        if len(options) < 3:
            return None

        random.shuffle(options)

        question = f"Who among {', '.join(options)} is NOT a child of {parent} AND NOT born before {year_threshold}?"
        answer = correct_answer

        metadata = {
            'question_type': 'multi_condition_not',
            'parent': parent,
            'year_threshold': year_threshold,
            'options': options,
            'correct_answer': correct_answer,
            'requires_multi_condition': True,
            'requires_temporal_logic': True
        }

        return (question, answer, 'hard', metadata)


if __name__ == '__main__':
    # Test the task generator
    from .data_loader import RoyalFamilyDataLoader

    print("Testing Task 6: Negative Reasoning")
    print("="*80)

    loader = RoyalFamilyDataLoader()
    task = NegativeReasoningTask(loader)

    # Generate sample questions
    print("\nGenerating 10 sample questions...")
    questions = task.generate_questions(count=10, difficulty='mixed')

    for i, q in enumerate(questions, 1):
        print(f"\n{i}. [{q['difficulty'].upper()}] - {q['metadata']['question_type']}")
        print(f"   Q: {q['question']}")
        print(f"   A: {q['answer']}")

    print(f"\n✓ Successfully generated {len(questions)} questions")
