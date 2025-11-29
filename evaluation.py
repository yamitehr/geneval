import json
import os
import time
from typing import Dict, List
from dotenv import load_dotenv
from google import genai

# Load environment variables from .env file
load_dotenv()


def load_questions(jsonl_path: str = 'data/examples_test.jsonl') -> List[Dict]:
    """Load questions from JSONL file"""
    questions = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            questions.append(json.loads(line))
    return questions


def get_llm_answer(client: genai.Client, question: str, model: str = "gemini-2.5-flash") -> str:
    """Get answer from Gemini LLM for a given question"""
    response = client.models.generate_content(
        model=model,
        contents=question,
    )
    return response.text.strip()


def compare_answers(client: genai.Client,
                   question: str,
                   llm_answer: str,
                   true_answer: str,
                   model: str = "gemini-2.0-flash-exp") -> Dict[str, any]:
    """
    Use LLM to compare if the LLM's answer matches the true answer.
    Returns dict with 'is_correct' boolean and 'explanation' string.
    """
    comparison_prompt = f"""You are evaluating whether an AI's answer to a genealogical question is correct.

Question: {question}
AI's Answer: {llm_answer}
Correct Answer: {true_answer}

Does the AI's answer correctly identify the same person as the correct answer?
Consider that answers might be formatted differently (e.g., with or without dates, with different titles).
The core identity must match.

Respond with ONLY a JSON object in this exact format:
{{
  "is_correct": true or false,
  "explanation": "Brief explanation of why the answers match or don't match"
}}"""

    response = client.models.generate_content(
        model=model,
        contents=comparison_prompt,
    )

    # Parse the JSON response
    response_text = response.text.strip()

    # Extract JSON from markdown code blocks if present
    if '```json' in response_text:
        response_text = response_text.split('```json')[1].split('```')[0].strip()
    elif '```' in response_text:
        response_text = response_text.split('```')[1].split('```')[0].strip()

    try:
        result = json.loads(response_text)
        return result
    except json.JSONDecodeError:
        # Fallback: try to parse manually
        return {
            "is_correct": False,
            "explanation": f"Failed to parse comparison result: {response_text}"
        }


def eval(jsonl_path: str = 'data/examples.jsonl', model: str = "gemini-2.0-flash-exp") -> Dict:
    """
    Evaluate LLM performance on genealogical questions.

    Args:
        jsonl_path: Path to JSONL file containing questions
        model: Gemini model to use for evaluation

    Returns:
        Dict containing evaluation results and statistics
    """
    # Initialize Gemini client with API key from environment
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment. Please set it in your .env file.")

    client = genai.Client(api_key=api_key)

    # Load questions
    questions = load_questions(jsonl_path)

    results = []
    correct_count = 0

    print(f"Evaluating {len(questions)} questions using {model}...\n")
    print("="*80)

    for i, q in enumerate(questions, 1):
        print(f"\n[Question {i}/{len(questions)}] ({q['difficulty'].upper()})")
        print(f"Q: {q['question']}")
        print(f"True Answer: {q['answer']}")

        # Get LLM's answer
        llm_answer = get_llm_answer(client, q['question'], model)
        print(f"LLM Answer: {llm_answer}")

        # Compare answers
        comparison = compare_answers(client, q['question'], llm_answer, q['answer'], model)
        is_correct = comparison['is_correct']

        if is_correct:
            correct_count += 1
            print(f"✓ CORRECT")
        else:
            print(f"✗ INCORRECT")

        print(f"Explanation: {comparison['explanation']}")

        # Store result
        results.append({
            'question': q['question'],
            'difficulty': q['difficulty'],
            'hops': q['hops'],
            'true_answer': q['answer'],
            'llm_answer': llm_answer,
            'is_correct': is_correct,
            'explanation': comparison['explanation']
        })

        # Rate limiting: space out requests to avoid hitting 10 RPM limit
        # Each question requires 2 API calls (answer + comparison)
        # For 10 RPM = 1 request per 6 seconds, wait 7 seconds between questions
        if i < len(questions):
            print(f"  [Rate limiting: waiting 7 seconds...]")
            time.sleep(7)

    # Calculate statistics
    accuracy = correct_count / len(questions) if questions else 0

    # Calculate accuracy by difficulty
    accuracy_by_difficulty = {}
    for difficulty in ['easy', 'medium', 'hard']:
        diff_questions = [r for r in results if r['difficulty'] == difficulty]
        if diff_questions:
            diff_correct = sum(1 for r in diff_questions if r['is_correct'])
            accuracy_by_difficulty[difficulty] = diff_correct / len(diff_questions)
        else:
            accuracy_by_difficulty[difficulty] = 0

    # Print summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"Total Questions: {len(questions)}")
    print(f"Correct: {correct_count}")
    print(f"Incorrect: {len(questions) - correct_count}")
    print(f"Overall Accuracy: {accuracy:.1%}")
    print(f"\nAccuracy by Difficulty:")
    for difficulty, acc in accuracy_by_difficulty.items():
        count = len([r for r in results if r['difficulty'] == difficulty])
        print(f"  {difficulty.capitalize()}: {acc:.1%} ({count} questions)")

    return {
        'model': model,
        'total_questions': len(questions),
        'correct': correct_count,
        'accuracy': accuracy,
        'accuracy_by_difficulty': accuracy_by_difficulty,
        'results': results
    }


if __name__ == '__main__':
    # Run evaluation
    eval_results = eval()

    # Optionally save results
    output_path = 'data/evaluation_results.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Saved detailed results to {output_path}")
