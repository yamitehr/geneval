"""
Base classes for GENEVAL task generators.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import random


class TaskGenerator(ABC):
    """
    Abstract base class for all task generators.
    Each task type inherits from this and implements generate_questions().
    """

    def __init__(self, data_loader, task_name: str, task_type: str):
        """
        Args:
            data_loader: RoyalFamilyDataLoader instance
            task_name: Human-readable task name (e.g., "Multi-Hop Relationship Traversal")
            task_type: Short identifier (e.g., "multi_hop")
        """
        self.data_loader = data_loader
        self.task_name = task_name
        self.task_type = task_type

    @abstractmethod
    def generate_questions(self, count: int, difficulty: str = 'mixed') -> List[Dict]:
        """
        Generate questions for this task.

        Args:
            count: Number of questions to generate
            difficulty: 'easy', 'medium', 'hard', or 'mixed'

        Returns:
            List of question dictionaries with structure:
            {
                'question': str,
                'answer': str,
                'task_type': str,
                'task_name': str,
                'difficulty': str,
                'metadata': dict  # Task-specific metadata
            }
        """
        pass

    def create_question_dict(self, question: str, answer: str, difficulty: str,
                            metadata: Optional[Dict] = None) -> Dict:
        """
        Create a standardized question dictionary.

        Args:
            question: The question text
            answer: The expected answer
            difficulty: 'easy', 'medium', or 'hard'
            metadata: Optional task-specific metadata

        Returns:
            Standardized question dictionary
        """
        return {
            'question': question,
            'answer': answer,
            'task_type': self.task_type,
            'task_name': self.task_name,
            'difficulty': difficulty,
            'metadata': metadata or {}
        }

    def pick_random_person(self, filter_fn=None) -> Optional[str]:
        """
        Pick a random person from the dataset.

        Args:
            filter_fn: Optional function to filter people (takes person_data dict)

        Returns:
            Canonical name of a random person, or None if no valid person found
        """
        candidates = list(self.data_loader.people.keys())

        if filter_fn:
            candidates = [
                name for name in candidates
                if filter_fn(self.data_loader.people[name])
            ]

        return random.choice(candidates) if candidates else None

    def pick_random_people(self, count: int, filter_fn=None, unique: bool = True) -> List[str]:
        """
        Pick multiple random people from the dataset.

        Args:
            count: Number of people to pick
            filter_fn: Optional function to filter people
            unique: If True, ensure all picked people are different

        Returns:
            List of canonical names
        """
        candidates = list(self.data_loader.people.keys())

        if filter_fn:
            candidates = [
                name for name in candidates
                if filter_fn(self.data_loader.people[name])
            ]

        if unique and count > len(candidates):
            count = len(candidates)

        if unique:
            return random.sample(candidates, count)
        else:
            return [random.choice(candidates) for _ in range(count)]
