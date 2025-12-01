"""
GENEVAL Results Analyzer
Extracts and presents detailed statistics from evaluation results.

Usage:
    python analyze_results.py --results data/evaluation_results.json --csv data/british_royal_family_FINAL.csv

Features:
    - Accuracy by royal house (Tudor, Stuart, Windsor, etc.)
    - Accuracy by time period (Medieval, Early Modern, Modern)
    - Accuracy by century (11th, 12th, 13th, etc.)
    - Accuracy by number of people involved
    - Accuracy by monarch involvement
    - Accuracy by gender composition
    - Accuracy by relationship type
"""

import json
import csv
import argparse
from collections import defaultdict
from typing import Dict, List, Set, Optional
import re
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()


# Common male and female first names for gender detection
MALE_NAMES = {
    'henry', 'edward', 'william', 'richard', 'john', 'charles', 'james', 'george',
    'stephen', 'harold', 'edmund', 'edgar', 'alfred', 'robert', 'thomas', 'philip',
    'humphrey', 'louis', 'david', 'peter', 'albert', 'frederick', 'arthur', 'leopold'
}

FEMALE_NAMES = {
    'mary', 'elizabeth', 'anne', 'margaret', 'victoria', 'matilda', 'eleanor',
    'catherine', 'jane', 'charlotte', 'sophia', 'emma', 'isabella', 'beatrice',
    'alice', 'alexandra', 'louise', 'diana', 'sarah', 'joan', 'cecily', 'philippa'
}


def load_csv_data(csv_path: str) -> Dict[str, Dict]:
    """
    Load CSV data and create lookup by canonical name.

    Returns:
        Dict mapping canonical_name -> person data
    """
    people = {}

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            canonical = row.get('canonical_name', '')
            if canonical:
                people[canonical] = row

    return people


def load_evaluation_results(results_path: str) -> Dict:
    """Load evaluation results JSON."""
    with open(results_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_canonical_names_from_text(text: str) -> List[str]:
    """
    Extract canonical names (format: Name (birth-death)) from text.

    Examples:
        "Henry VIII (1491-1547)"
        "Elizabeth I (1533-1603)"
    """
    # Pattern: Name (YYYY-YYYY) or Name (YYYY-?)
    # Note: Accept both en-dash (–) and hyphen (-) in date ranges
    pattern = r'([A-Z][a-zA-Z\s,\.]+(?:\s+[IVX]+)?)\s*\((\d{4}|\?)[–-](\d{4}|\?)\)'
    matches = re.findall(pattern, text)

    canonical_names = []
    for name, birth, death in matches:
        name = name.strip()
        # Use en-dash (–) to match CSV canonical_name format
        canonical = f"{name} ({birth}–{death})"
        canonical_names.append(canonical)

    return canonical_names


def get_gender_from_name(first_name: str) -> Optional[str]:
    """Determine gender from first name."""
    first_name_lower = first_name.lower().strip()

    if first_name_lower in MALE_NAMES:
        return 'male'
    elif first_name_lower in FEMALE_NAMES:
        return 'female'

    return None


def get_gender_from_person_data(person_data: Dict) -> Optional[str]:
    """
    Get gender from person data (CSV field).

    The CSV has a 'gender' column populated from Wikidata.
    Falls back to name-based inference if CSV field is missing.
    """
    # Prefer CSV gender field (populated from Wikidata)
    csv_gender = person_data.get('gender', '').strip().lower()
    if csv_gender in ('male', 'female'):
        return csv_gender

    # Fallback: try first name matching
    first_name = person_data.get('first_name', '')
    gender = get_gender_from_name(first_name)
    if gender:
        return gender

    return None


def get_time_period(birth_year: str) -> str:
    """Determine time period from birth year."""
    try:
        year = int(birth_year)
        if year < 1500:
            return 'Medieval'
        elif year < 1700:
            return 'Early Modern'
        else:
            return 'Modern'
    except (ValueError, TypeError):
        return 'Unknown'


def get_century(birth_year: str) -> str:
    """Determine century from birth year."""
    try:
        year = int(birth_year)
        century = ((year - 1) // 100) + 1
        return f"{century}th century"
    except (ValueError, TypeError):
        return 'Unknown'


def is_monarch(person_data: Dict) -> bool:
    """Check if person was a monarch (has reign_start)."""
    reign_start = person_data.get('reign_start', '').strip()
    return bool(reign_start)


def extract_question_metadata(question_text: str,
                               true_answer: str,
                               metadata: Dict,
                               people_lookup: Dict[str, Dict],
                               task_type: str = '') -> Dict:
    """
    Extract comprehensive metadata for a question.

    Returns:
        Dict with keys: royal_houses, time_periods, centuries, num_people,
                       monarch_involvement, gender_composition, relationship_type
    """
    # Extract canonical names from question and answer
    canonical_names = extract_canonical_names_from_text(question_text)
    canonical_names.extend(extract_canonical_names_from_text(true_answer))
    canonical_names = list(set(canonical_names))  # Remove duplicates

    # Get person data for all involved people
    people_data = []
    for name in canonical_names:
        if name in people_lookup:
            people_data.append(people_lookup[name])

    # If no canonical names found, try to get from metadata
    if not people_data and 'people' in metadata:
        for person_name in metadata['people']:
            if person_name in people_lookup:
                people_data.append(people_lookup[person_name])

    # Extract royal houses
    royal_houses = set()
    for person in people_data:
        house = person.get('house', '').strip()
        if house:
            royal_houses.add(house)

    # Extract time periods and centuries
    time_periods = set()
    centuries = set()
    for person in people_data:
        birth_date = person.get('date_of_birth', '').strip()
        if birth_date:
            birth_year = birth_date.split('-')[0]  # Extract year from YYYY-MM-DD
            time_periods.add(get_time_period(birth_year))
            centuries.add(get_century(birth_year))

    # Monarch involvement
    monarchs = [p for p in people_data if is_monarch(p)]
    if len(monarchs) == len(people_data) and people_data:
        monarch_involvement = 'all_monarchs'
    elif monarchs:
        monarch_involvement = 'has_monarch'
    else:
        monarch_involvement = 'no_monarchs'

    # Gender composition
    genders = []
    for person in people_data:
        gender = get_gender_from_person_data(person)
        if gender:
            genders.append(gender)

    male_count = genders.count('male')
    female_count = genders.count('female')

    if male_count > 0 and female_count > 0:
        gender_composition = 'mixed'
    elif male_count > 0 and female_count == 0:
        gender_composition = 'all_male'
    elif female_count > 0 and male_count == 0:
        gender_composition = 'all_female'
    else:
        gender_composition = 'unknown'

    # Determine relationship type from metadata
    relationship_type = 'unknown'
    if 'path' in metadata:
        path = metadata['path']
        if 'spouse' in path:
            relationship_type = 'spouse'
        elif 'sibling' in ' '.join(path).lower():
            relationship_type = 'sibling'
        elif any(rel in path for rel in ['father', 'mother', 'parent']):
            if len(path) == 1:
                relationship_type = 'parent-child'
            else:
                relationship_type = 'multi-generational'
    elif 'question_type' in metadata:
        qt = metadata['question_type']
        if 'sibling' in qt:
            relationship_type = 'sibling'
        elif 'parent' in qt or 'child' in qt:
            relationship_type = 'parent-child'
        elif 'age' in qt or 'death' in qt or 'lifespan' in qt:
            relationship_type = 'temporal'
        elif 'name' in qt and 'relationship' in qt:
            relationship_type = 'disambiguation'

    # Fallback: use task_type if relationship still unknown
    if relationship_type == 'unknown' and task_type:
        if 'multi_hop' in task_type:
            relationship_type = 'multi-generational'
        elif 'temporal' in task_type:
            relationship_type = 'temporal'
        elif 'constraint' in task_type:
            relationship_type = 'constraint'
        elif 'ambiguity' in task_type:
            relationship_type = 'disambiguation'
        elif 'degradation' in task_type:
            relationship_type = 'multi-generational'
        elif 'negative' in task_type:
            relationship_type = 'negative'
        elif 'sibling' in task_type:
            relationship_type = 'sibling'
        elif 'lifespan' in task_type:
            relationship_type = 'temporal'

    return {
        'royal_houses': list(royal_houses),
        'time_periods': list(time_periods),
        'centuries': list(centuries),
        'num_people': len(people_data),
        'monarch_involvement': monarch_involvement,
        'gender_composition': gender_composition,
        'relationship_type': relationship_type,
        'has_cross_dynasty': len(royal_houses) > 1
    }


def calculate_segmentation_stats(results: List[Dict],
                                  people_lookup: Dict[str, Dict]) -> Dict:
    """
    Calculate accuracy statistics for all segmentations.

    Returns:
        Dict with statistics for each segmentation type
    """
    # Initialize counters
    stats = {
        'by_house': defaultdict(lambda: {'total': 0, 'correct': 0}),
        'by_time_period': defaultdict(lambda: {'total': 0, 'correct': 0}),
        'by_century': defaultdict(lambda: {'total': 0, 'correct': 0}),
        'by_num_people': defaultdict(lambda: {'total': 0, 'correct': 0}),
        'by_monarch': defaultdict(lambda: {'total': 0, 'correct': 0}),
        'by_gender': defaultdict(lambda: {'total': 0, 'correct': 0}),
        'by_relationship': defaultdict(lambda: {'total': 0, 'correct': 0}),
        'by_cross_dynasty': defaultdict(lambda: {'total': 0, 'correct': 0}),
        'by_hop_count': defaultdict(lambda: {'total': 0, 'correct': 0}),
    }

    # Process each result
    for result in results:
        question = result['question']
        true_answer = result['true_answer']
        metadata = result.get('metadata', {})
        is_correct = result['is_correct']
        task_type = result.get('task_type', '')

        # Extract metadata
        q_meta = extract_question_metadata(question, true_answer, metadata, people_lookup, task_type)

        # Update royal house stats
        for house in q_meta['royal_houses']:
            stats['by_house'][house]['total'] += 1
            if is_correct:
                stats['by_house'][house]['correct'] += 1

        # If question involves multiple houses, also count it
        if q_meta['has_cross_dynasty']:
            key = 'cross_dynasty'
            stats['by_cross_dynasty'][key]['total'] += 1
            if is_correct:
                stats['by_cross_dynasty'][key]['correct'] += 1
        else:
            key = 'single_dynasty'
            stats['by_cross_dynasty'][key]['total'] += 1
            if is_correct:
                stats['by_cross_dynasty'][key]['correct'] += 1

        # Update time period stats
        for period in q_meta['time_periods']:
            stats['by_time_period'][period]['total'] += 1
            if is_correct:
                stats['by_time_period'][period]['correct'] += 1

        # Update century stats
        for century in q_meta['centuries']:
            stats['by_century'][century]['total'] += 1
            if is_correct:
                stats['by_century'][century]['correct'] += 1

        # Update number of people stats
        if q_meta['num_people'] > 0:
            num_people_key = str(q_meta['num_people'])
            stats['by_num_people'][num_people_key]['total'] += 1
            if is_correct:
                stats['by_num_people'][num_people_key]['correct'] += 1

        # Update monarch involvement stats
        stats['by_monarch'][q_meta['monarch_involvement']]['total'] += 1
        if is_correct:
            stats['by_monarch'][q_meta['monarch_involvement']]['correct'] += 1

        # Update gender stats
        if q_meta['gender_composition'] != 'unknown':
            stats['by_gender'][q_meta['gender_composition']]['total'] += 1
            if is_correct:
                stats['by_gender'][q_meta['gender_composition']]['correct'] += 1

        # Update relationship type stats
        stats['by_relationship'][q_meta['relationship_type']]['total'] += 1
        if is_correct:
            stats['by_relationship'][q_meta['relationship_type']]['correct'] += 1

        # Update hop count stats (for multi-hop questions only)
        hops = metadata.get('hops')
        if hops is not None:
            hop_key = str(hops)
            stats['by_hop_count'][hop_key]['total'] += 1
            if is_correct:
                stats['by_hop_count'][hop_key]['correct'] += 1

    # Calculate accuracies
    for seg_type in stats:
        for key in stats[seg_type]:
            total = stats[seg_type][key]['total']
            correct = stats[seg_type][key]['correct']
            stats[seg_type][key]['accuracy'] = correct / total if total > 0 else 0

    return stats


def create_stats_table(title: str, stats_dict: Dict, sort_key=None) -> Table:
    """Helper to create a statistics table."""
    table = Table(title=f"[bold cyan]{title}[/bold cyan]", box=box.ROUNDED, border_style="cyan")
    table.add_column("Category", style="white", width=45)
    table.add_column("Accuracy", justify="right", style="yellow", width=10)
    table.add_column("Correct/Total", justify="center", style="dim", width=12)

    items = sorted(stats_dict.items(), key=sort_key) if sort_key else stats_dict.items()

    for category, stats in items:
        acc = stats['accuracy']
        total = stats['total']
        correct = stats['correct']
        acc_color = "green" if acc >= 0.8 else "yellow" if acc >= 0.6 else "red"
        table.add_row(
            category,
            f"[{acc_color}]{acc:.1%}[/{acc_color}]",
            f"{correct}/{total}"
        )

    return table


def print_statistics(eval_results: Dict, segmentation_stats: Dict):
    """Print comprehensive statistics in a single consolidated table."""

    # Header
    console.print("\n")
    console.print(Panel.fit(
        f"[bold cyan]Evaluation Results: {eval_results.get('model', 'N/A')} | "
        f"Overall Accuracy: {eval_results.get('overall_accuracy', 0):.1%}[/bold cyan]",
        border_style="cyan"
    ))

    # Create one comprehensive table
    console.print("\n")
    main_table = Table(
        title=f"[bold cyan]Complete Analysis ({eval_results.get('total_questions', 0)} questions)[/bold cyan]",
        box=box.ROUNDED,
        border_style="cyan",
        show_lines=True
    )

    main_table.add_column("Segmentation", style="cyan bold", width=25)
    main_table.add_column("Category", style="white", width=40)
    main_table.add_column("Accuracy", justify="right", style="yellow", width=10)
    main_table.add_column("Correct/Total", justify="center", style="dim", width=12)

    # Add Task Type section
    main_table.add_row("[bold magenta]TASK TYPE[/bold magenta]", "", "", "", style="on grey23")
    for task_name, task_stats in sorted(eval_results.get('task_statistics', {}).items()):
        acc = task_stats['accuracy']
        acc_color = "green" if acc >= 0.8 else "yellow" if acc >= 0.6 else "red"
        main_table.add_row(
            "Task Type",
            task_name,
            f"[{acc_color}]{acc:.1%}[/{acc_color}]",
            f"{task_stats['correct']}/{task_stats['total']}"
        )

    # Add Difficulty section
    main_table.add_row("[bold magenta]DIFFICULTY[/bold magenta]", "", "", "", style="on grey23")
    for difficulty in ['easy', 'medium', 'hard']:
        if difficulty in eval_results.get('difficulty_statistics', {}):
            diff_stats = eval_results['difficulty_statistics'][difficulty]
            acc = diff_stats['accuracy']
            acc_color = "green" if acc >= 0.8 else "yellow" if acc >= 0.6 else "red"
            main_table.add_row(
                "Difficulty",
                difficulty.capitalize(),
                f"[{acc_color}]{acc:.1%}[/{acc_color}]",
                f"{diff_stats['correct']}/{diff_stats['total']}"
            )

    # Add Royal House section
    main_table.add_row("[bold magenta]ROYAL HOUSE[/bold magenta]", "", "", "", style="on grey23")
    for house in sorted(segmentation_stats['by_house'].keys()):
        stats = segmentation_stats['by_house'][house]
        acc = stats['accuracy']
        acc_color = "green" if acc >= 0.8 else "yellow" if acc >= 0.6 else "red"
        main_table.add_row(
            "Royal House",
            house,
            f"[{acc_color}]{acc:.1%}[/{acc_color}]",
            f"{stats['correct']}/{stats['total']}"
        )

    # Add Time Period section
    main_table.add_row("[bold magenta]TIME PERIOD[/bold magenta]", "", "", "", style="on grey23")
    period_order = {'Medieval': 1, 'Early Modern': 2, 'Modern': 3, 'Unknown': 4}
    for period in sorted(segmentation_stats['by_time_period'].keys(), key=lambda x: period_order.get(x, 999)):
        stats = segmentation_stats['by_time_period'][period]
        acc = stats['accuracy']
        acc_color = "green" if acc >= 0.8 else "yellow" if acc >= 0.6 else "red"
        main_table.add_row(
            "Time Period",
            period,
            f"[{acc_color}]{acc:.1%}[/{acc_color}]",
            f"{stats['correct']}/{stats['total']}"
        )

    # Add Century section
    main_table.add_row("[bold magenta]CENTURY (TOP 5)[/bold magenta]", "", "", "", style="on grey23")
    century_sort = lambda x: int(x.split('th')[0]) if x != 'Unknown' else 999
    top_centuries = sorted(segmentation_stats['by_century'].keys(), key=century_sort)[:5]
    for century in top_centuries:
        stats = segmentation_stats['by_century'][century]
        acc = stats['accuracy']
        acc_color = "green" if acc >= 0.8 else "yellow" if acc >= 0.6 else "red"
        main_table.add_row(
            "Century (top 5)",
            century,
            f"[{acc_color}]{acc:.1%}[/{acc_color}]",
            f"{stats['correct']}/{stats['total']}"
        )

    # Add Number of People section
    main_table.add_row("[bold magenta]PEOPLE INVOLVED[/bold magenta]", "", "", "", style="on grey23")
    for num in sorted(segmentation_stats['by_num_people'].keys(), key=lambda x: int(x) if x.isdigit() else 999):
        stats = segmentation_stats['by_num_people'][num]
        label = f"{num} {'person' if num == '1' else 'people'}" if num.isdigit() else num
        acc = stats['accuracy']
        acc_color = "green" if acc >= 0.8 else "yellow" if acc >= 0.6 else "red"
        main_table.add_row(
            "People Involved",
            label,
            f"[{acc_color}]{acc:.1%}[/{acc_color}]",
            f"{stats['correct']}/{stats['total']}"
        )

    # Add Monarch Involvement section
    main_table.add_row("[bold magenta]MONARCH[/bold magenta]", "", "", "", style="on grey23")
    monarch_labels = {
        'all_monarchs': 'All monarchs',
        'has_monarch': 'Has monarch',
        'no_monarchs': 'No monarchs'
    }
    for key in ['all_monarchs', 'has_monarch', 'no_monarchs']:
        if key in segmentation_stats['by_monarch']:
            stats = segmentation_stats['by_monarch'][key]
            acc = stats['accuracy']
            acc_color = "green" if acc >= 0.8 else "yellow" if acc >= 0.6 else "red"
            main_table.add_row(
                "Monarch",
                monarch_labels[key],
                f"[{acc_color}]{acc:.1%}[/{acc_color}]",
                f"{stats['correct']}/{stats['total']}"
            )

    # Add Gender Composition section
    main_table.add_row("[bold magenta]GENDER[/bold magenta]", "", "", "", style="on grey23")
    gender_labels = {
        'all_male': 'All males',
        'all_female': 'All females',
        'mixed': 'Mixed',
        'unknown': 'Unknown'
    }
    for key in ['all_male', 'all_female', 'mixed', 'unknown']:
        if key in segmentation_stats['by_gender']:
            stats = segmentation_stats['by_gender'][key]
            acc = stats['accuracy']
            acc_color = "green" if acc >= 0.8 else "yellow" if acc >= 0.6 else "red"
            main_table.add_row(
                "Gender",
                gender_labels[key],
                f"[{acc_color}]{acc:.1%}[/{acc_color}]",
                f"{stats['correct']}/{stats['total']}"
            )

    # Add Relationship Type section
    main_table.add_row("[bold magenta]RELATIONSHIP[/bold magenta]", "", "", "", style="on grey23")
    for rel_type in sorted(segmentation_stats['by_relationship'].keys()):
        stats = segmentation_stats['by_relationship'][rel_type]
        acc = stats['accuracy']
        acc_color = "green" if acc >= 0.8 else "yellow" if acc >= 0.6 else "red"
        main_table.add_row(
            "Relationship",
            rel_type.replace('_', ' ').title(),
            f"[{acc_color}]{acc:.1%}[/{acc_color}]",
            f"{stats['correct']}/{stats['total']}"
        )

    # Add Dynasty Crossing section
    main_table.add_row("[bold magenta]DYNASTY[/bold magenta]", "", "", "", style="on grey23")
    cross_labels = {
        'single_dynasty': 'Single house',
        'cross_dynasty': 'Cross-house'
    }
    for key in ['single_dynasty', 'cross_dynasty']:
        if key in segmentation_stats['by_cross_dynasty']:
            stats = segmentation_stats['by_cross_dynasty'][key]
            acc = stats['accuracy']
            acc_color = "green" if acc >= 0.8 else "yellow" if acc >= 0.6 else "red"
            main_table.add_row(
                "Dynasty",
                cross_labels[key],
                f"[{acc_color}]{acc:.1%}[/{acc_color}]",
                f"{stats['correct']}/{stats['total']}"
            )

    # Add Hop Count section (Compositional Degradation Curve)
    if segmentation_stats['by_hop_count']:
        main_table.add_row("[bold magenta]HOP COUNT (Compositional Degradation)[/bold magenta]", "", "", "", style="on grey23")
        for hops in sorted(segmentation_stats['by_hop_count'].keys(), key=lambda x: int(x)):
            stats = segmentation_stats['by_hop_count'][hops]
            acc = stats['accuracy']
            acc_color = "green" if acc >= 0.8 else "yellow" if acc >= 0.6 else "red"
            main_table.add_row(
                "Hop Count",
                f"{hops} hops",
                f"[{acc_color}]{acc:.1%}[/{acc_color}]",
                f"{stats['correct']}/{stats['total']}"
            )

    console.print(main_table)

    # Footer
    console.print("\n")
    console.print(Panel.fit("[bold green]✓ Analysis Complete![/bold green]", border_style="green"))


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Analyze GENEVAL evaluation results with detailed segmentations'
    )
    parser.add_argument(
        '--results',
        type=str,
        required=True,
        help='Path to evaluation results JSON file'
    )
    parser.add_argument(
        '--csv',
        type=str,
        default='british_royal_family_FINAL.csv',
        help='Path to British Royal Family CSV (default: british_royal_family_FINAL.csv)'
    )

    args = parser.parse_args()

    console.print("\n[cyan]Loading data...[/cyan]")
    people_lookup = load_csv_data(args.csv)
    console.print(f"[green]✓[/green] Loaded {len(people_lookup)} people from CSV")

    eval_results = load_evaluation_results(args.results)
    console.print(f"[green]✓[/green] Loaded evaluation results ({eval_results.get('total_questions', 0)} questions)")

    console.print("\n[cyan]Calculating segmentation statistics...[/cyan]")
    results_list = eval_results.get('results', [])
    segmentation_stats = calculate_segmentation_stats(results_list, people_lookup)
    console.print("[green]✓[/green] Statistics calculated")

    print_statistics(eval_results, segmentation_stats)


if __name__ == '__main__':
    main()