"""
Data loader for British Royal Family genealogical data.
Computes derived relationships and temporal information for GENEVAL benchmark.
"""

import csv
from datetime import datetime
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict


class RoyalFamilyDataLoader:
    """
    Data loader that computes:
    - Parent-child relationships (reverse mappings)
    - Sibling relationships (shared parents)
    - Generation distances
    - Temporal information (lifespans, ages, reign periods)
    - House/dynasty memberships
    """

    def __init__(self, csv_path='british_royal_family_FINAL.csv'):
        self.csv_path = csv_path
        self.people = {}  # canonical_name -> person_data
        self.parent_to_children = defaultdict(list)  # parent -> list of children
        self.siblings = defaultdict(set)  # person -> set of siblings
        self.generation_cache = {}  # (person1, person2) -> generation_distance

        # Load and process data
        self.load_data()
        self.build_parent_child_mappings()
        self.build_sibling_relationships()
        self.compute_temporal_data()
        self.index_by_house()

    def load_data(self):
        """Load royal family data from CSV file"""
        print("Loading data from CSV...")
        with open(self.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            for row in reader:
                canonical_name = row['canonical_name']

                # Skip empty entries
                if not canonical_name or '(' not in canonical_name:
                    continue

                # Store this person's data
                self.people[canonical_name] = {
                    'canonical_name': canonical_name,
                    'first_name': row['first_name'],
                    'last_name': row['last_name'],
                    'date_of_birth': row['date_of_birth'],
                    'date_of_death': row['date_of_death'],
                    'mother': row['mother'],
                    'father': row['father'],
                    'reign_start': row['reign_start'],
                    'reign_end': row['reign_end'],
                    'house': row['house'],
                    'spouse': row['spouse'],
                    'wikidata_uri': row['wikidata_uri']
                }

        print(f"✓ Loaded {len(self.people)} people")

    def find_person(self, name: str) -> Optional[str]:
        """
        Find a person's canonical name from various name formats.
        Returns canonical_name or None if not found.
        """
        if not name:
            return None

        name = name.strip()

        # Try exact match first
        if name in self.people:
            return name

        # Try partial match on name part before dates
        for canonical_name in self.people.keys():
            name_part = canonical_name.split('(')[0].strip()

            # Check various matching strategies
            if (name.lower() == name_part.lower() or
                name.lower().replace(',', '') == name_part.lower().replace(',', '')):
                return canonical_name

        return None

    def build_parent_child_mappings(self):
        """Build reverse mappings: parent -> list of children"""
        print("\nBuilding parent-child mappings...")

        for person_name, person_data in self.people.items():
            mother_name = person_data['mother']
            father_name = person_data['father']

            # Find canonical names for parents
            mother_canonical = self.find_person(mother_name) if mother_name else None
            father_canonical = self.find_person(father_name) if father_name else None

            # Add to parent->children mapping
            if mother_canonical:
                self.parent_to_children[mother_canonical].append(person_name)
            if father_canonical:
                self.parent_to_children[father_canonical].append(person_name)

        total_parents = len(self.parent_to_children)
        total_parent_child_links = sum(len(children) for children in self.parent_to_children.values())
        print(f"✓ Found {total_parents} people with children")
        print(f"✓ Total parent-child relationships: {total_parent_child_links}")

    def build_sibling_relationships(self):
        """Build sibling relationships based on shared parents"""
        print("\nBuilding sibling relationships...")

        for person_name, person_data in self.people.items():
            mother = self.find_person(person_data['mother'])
            father = self.find_person(person_data['father'])

            # Find all people who share at least one parent
            for other_name, other_data in self.people.items():
                if other_name == person_name:
                    continue

                other_mother = self.find_person(other_data['mother'])
                other_father = self.find_person(other_data['father'])

                # Check for shared parents
                if mother and mother == other_mother:
                    self.siblings[person_name].add(other_name)
                elif father and father == other_father:
                    self.siblings[person_name].add(other_name)

        people_with_siblings = len([p for p in self.siblings if len(self.siblings[p]) > 0])
        total_sibling_pairs = sum(len(sibs) for sibs in self.siblings.values()) // 2
        print(f"✓ Found {people_with_siblings} people with siblings")
        print(f"✓ Total sibling pairs: {total_sibling_pairs}")

    def parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date string to datetime object"""
        if not date_str:
            return None

        try:
            return datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError:
            return None

    def compute_lifespan(self, person_name: str) -> Optional[int]:
        """Compute lifespan in years for a person"""
        person = self.people.get(person_name)
        if not person:
            return None

        birth = self.parse_date(person['date_of_birth'])
        death = self.parse_date(person['date_of_death'])

        if birth and death:
            return (death - birth).days // 365
        return None

    def compute_temporal_data(self):
        """Compute temporal information for all people"""
        print("\nComputing temporal data (lifespans, ages, reign periods)...")

        for person_name, person_data in self.people.items():
            birth = self.parse_date(person_data['date_of_birth'])
            death = self.parse_date(person_data['date_of_death'])

            # Compute lifespan
            lifespan = None
            if birth and death:
                lifespan = (death - birth).days // 365

            # Compute age at coronation (if monarch)
            age_at_coronation = None
            if birth and person_data['reign_start']:
                reign_start = self.parse_date(person_data['reign_start'])
                if reign_start:
                    age_at_coronation = (reign_start - birth).days // 365

            # Add computed fields
            person_data['lifespan_years'] = lifespan
            person_data['age_at_coronation'] = age_at_coronation

        people_with_lifespans = len([p for p in self.people.values() if p['lifespan_years'] is not None])
        people_with_coronation_age = len([p for p in self.people.values() if p['age_at_coronation'] is not None])

        print(f"✓ Computed lifespans for {people_with_lifespans} people")
        print(f"✓ Computed coronation ages for {people_with_coronation_age} monarchs")

    def index_by_house(self):
        """Build index of people by royal house"""
        self.houses = defaultdict(list)

        for person_name, person_data in self.people.items():
            house = person_data.get('house', '').strip()
            if house:
                self.houses[house].append(person_name)

        print(f"\n✓ Found {len(self.houses)} royal houses")
        for house, members in sorted(self.houses.items(), key=lambda x: len(x[1]), reverse=True):
            print(f"  {house}: {len(members)} members")

    def get_children(self, person_name: str) -> List[str]:
        """Get list of children for a person"""
        return self.parent_to_children.get(person_name, [])

    def get_siblings(self, person_name: str, full_siblings_only: bool = False) -> Set[str]:
        """
        Get siblings of a person.
        If full_siblings_only=True, only returns siblings with both parents shared.
        """
        if not full_siblings_only:
            return self.siblings.get(person_name, set())

        # Filter for full siblings (both parents match)
        person = self.people.get(person_name)
        if not person:
            return set()

        mother = self.find_person(person['mother'])
        father = self.find_person(person['father'])

        if not (mother and father):
            return set()

        full_sibs = set()
        for sib_name in self.siblings.get(person_name, set()):
            sib = self.people.get(sib_name)
            if sib:
                sib_mother = self.find_person(sib['mother'])
                sib_father = self.find_person(sib['father'])
                if sib_mother == mother and sib_father == father:
                    full_sibs.add(sib_name)

        return full_sibs

    def get_parents(self, person_name: str) -> Tuple[Optional[str], Optional[str]]:
        """Get (mother, father) tuple for a person"""
        person = self.people.get(person_name)
        if not person:
            return None, None

        mother = self.find_person(person['mother'])
        father = self.find_person(person['father'])
        return mother, father

    def has_both_parents_in_dataset(self, person_name: str) -> bool:
        """
        Check if both parents of a person exist in the dataset.
        Returns True only if BOTH mother and father are found.
        This is critical for verifiable "full sibling" questions.
        """
        mother, father = self.get_parents(person_name)
        return mother is not None and father is not None

    def compute_generation_distance(self, person1: str, person2: str) -> Optional[int]:
        """
        Compute generation distance between two people.
        Returns number of generations separating them, or None if not related.
        Positive if person2 is descendant of person1, negative if ancestor.
        """
        # Check cache first
        cache_key = (person1, person2)
        if cache_key in self.generation_cache:
            return self.generation_cache[cache_key]

        # BFS to find shortest path
        from collections import deque

        # Search upward from person1
        visited_up = {person1: 0}
        queue = deque([(person1, 0)])

        while queue:
            current, dist = queue.popleft()

            if current == person2:
                self.generation_cache[cache_key] = -dist
                return -dist

            mother, father = self.get_parents(current)
            for parent in [mother, father]:
                if parent and parent not in visited_up:
                    visited_up[parent] = dist + 1
                    queue.append((parent, dist + 1))

        # Search downward from person1
        visited_down = {person1: 0}
        queue = deque([(person1, 0)])

        while queue:
            current, dist = queue.popleft()

            if current == person2:
                self.generation_cache[cache_key] = dist
                return dist

            for child in self.get_children(current):
                if child not in visited_down:
                    visited_down[child] = dist + 1
                    queue.append((child, dist + 1))

        # Not directly related
        self.generation_cache[cache_key] = None
        return None

    def get_summary_stats(self):
        """Print summary statistics about the dataset"""
        print("\n" + "="*80)
        print("DATASET SUMMARY")
        print("="*80)
        print(f"Total people: {len(self.people)}")
        print(f"People with known mother: {sum(1 for p in self.people.values() if p['mother'])}")
        print(f"People with known father: {sum(1 for p in self.people.values() if p['father'])}")
        print(f"People with both parents: {sum(1 for p in self.people.values() if p['mother'] and p['father'])}")
        print(f"People with spouse data: {sum(1 for p in self.people.values() if p['spouse'])}")
        print(f"Monarchs (with reign data): {sum(1 for p in self.people.values() if p['reign_start'])}")
        print(f"People with house data: {sum(1 for p in self.people.values() if p['house'])}")
        print(f"People with known lifespan: {sum(1 for p in self.people.values() if p.get('lifespan_years'))}")
        print(f"\nParents with children: {len(self.parent_to_children)}")
        print(f"People with siblings: {len([p for p in self.siblings if len(self.siblings[p]) > 0])}")
        print("="*80)


if __name__ == '__main__':
    # Test the data loader
    loader = RoyalFamilyDataLoader()
    loader.get_summary_stats()

    # Test some functions
    print("\n" + "="*80)
    print("TESTING FUNCTIONS")
    print("="*80)

    # Test children lookup
    test_person = "Henry I of England (1068–1135)"
    children = loader.get_children(test_person)
    print(f"\n{test_person} has {len(children)} children:")
    for child in children[:5]:
        print(f"  - {child}")
    if len(children) > 5:
        print(f"  ... and {len(children) - 5} more")

    # Test sibling lookup
    test_person = "Edward III of England (1312–1377)"
    siblings = loader.get_siblings(test_person)
    print(f"\n{test_person} has {len(siblings)} siblings:")
    for sib in list(siblings)[:5]:
        print(f"  - {sib}")

    # Test generation distance
    person1 = "Edward III of England (1312–1377)"
    person2 = "Henry III of England (1207–1272)"
    dist = loader.compute_generation_distance(person1, person2)
    print(f"\nGeneration distance from {person1} to {person2}: {dist}")
