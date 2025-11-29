"""
GENEVAL Evaluation Runner
Runs LLM evaluation on the generated dataset and computes comprehensive statistics.

Usage:
    python run_eval.py [--data DATA_FILE] [--model MODEL_NAME] [--output OUTPUT_FILE]

Example:
    python run_eval.py --data data/examples.jsonl --model gemini-2.0-flash-exp
"""

import json
import os
import time
import argparse
from collections import defaultdict
from typing import Dict, List
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables
load_dotenv()

# Model-specific thinking budget ranges
# -1 = Dynamic thinking (model decides when and how much to think)
# 0 = Disable thinking (only for Flash, not Pro)
MODEL_THINKING_BUDGET_RANGE = {
    'gemini-2.5-flash': {'min': 0, 'max': 24576, 'default': -1},
    'gemini-2.5-pro': {'min': 128, 'max': 32768, 'default': -1},
}


def get_thinking_budget(model: str) -> int:
    """
    Get thinking budget from environment or use model-specific default.

    Special values:
    - -1: Dynamic thinking (model decides when and how much to think)
    - 0: Disable thinking (only for Flash, not Pro)

    Args:
        model: The Gemini model name

    Returns:
        Thinking budget (number of thinking tokens)
    """
    # Try to get from environment variable
    env_budget = os.getenv('THINKING_BUDGET')
    if env_budget:
        try:
            budget = int(env_budget)
        except ValueError:
            print(f"Warning: Invalid THINKING_BUDGET value '{env_budget}', using model default")
            budget = None
    else:
        budget = None

    # Get model-specific range
    model_config = MODEL_THINKING_BUDGET_RANGE.get(model)

    # If no env var set or invalid, use model default
    if budget is None:
        if model_config:
            return model_config['default']
        else:
            # Unknown model, use dynamic thinking
            print(f"Warning: Unknown model {model}, using dynamic thinking (-1)")
            return -1

    # Special values
    if budget == -1:
        return -1  # Dynamic thinking
    if budget == 0:
        if model_config and model_config['min'] == 0:
            return 0  # Disable thinking (only for Flash)
        else:
            print(f"Warning: Cannot disable thinking for {model}, using minimum ({model_config['min']})")
            return model_config['min']

    # Enforce range if model is known
    if model_config:
        if budget < model_config['min']:
            print(f"Warning: THINKING_BUDGET ({budget}) is below minimum for {model} ({model_config['min']}), using minimum")
            return model_config['min']
        if budget > model_config['max']:
            print(f"Warning: THINKING_BUDGET ({budget}) exceeds maximum for {model} ({model_config['max']}), using maximum")
            return model_config['max']

    return budget


def load_questions(jsonl_path: str) -> List[Dict]:
    """Load questions from JSONL file"""
    questions = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            questions.append(json.loads(line))
    return questions


def get_llm_answer(client: genai.Client, question: str, model: str, thinking_budget: int) -> str:
    """Get answer from LLM for a given question"""
    try:
        response = client.models.generate_content(
            model=model,
            contents=question,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget)
            ),
        )
        return response.text.strip()
    except Exception as e:
        return f"ERROR: {str(e)}"


def prepare_batch_requests(questions: List[Dict], thinking_budget: int) -> List[Dict]:
    """
    Convert questions to batch API format (inline requests).

    Args:
        questions: List of question dicts
        thinking_budget: Thinking budget for each request

    Returns:
        List of batch request dicts (no key/request wrapper for inline)
    """
    batch_requests = []

    for q in questions:
        # For inline requests, just pass the raw request object
        # The key/request wrapper is only for JSONL files
        request = {
            'contents': [{
                'parts': [{'text': q['question']}],
                'role': 'user'
            }]
        }

        # Add thinking config if budget is > 0
        if thinking_budget > 0:
            request['config'] = {
                'thinking_config': {
                    'thinking_budget': thinking_budget
                }
            }

        batch_requests.append(request)

    return batch_requests


def poll_batch_completion(client: genai.Client, job_name: str, poll_interval: int = 30) -> Dict:
    """
    Poll batch job until completion.

    Args:
        client: Gemini client
        job_name: Batch job name
        poll_interval: Seconds between polls

    Returns:
        Completed batch job
    """
    completed_states = {
        'JOB_STATE_SUCCEEDED',
        'JOB_STATE_FAILED',
        'JOB_STATE_CANCELLED',
        'JOB_STATE_EXPIRED',
    }

    print(f"\nPolling batch job: {job_name}")
    print(f"Checking status every {poll_interval} seconds...")

    start_time = time.time()
    job = client.batches.get(name=job_name)

    while job.state.name not in completed_states:
        elapsed = time.time() - start_time
        print(f"  Status: {job.state.name} (elapsed: {elapsed:.0f}s)")

        time.sleep(poll_interval)
        job = client.batches.get(name=job_name)

    elapsed = time.time() - start_time
    print(f"\n✓ Job completed in {elapsed:.0f}s")
    print(f"  Final state: {job.state.name}")

    if hasattr(job, 'stats'):
        print(f"  Total requests: {job.stats.total_request_count}")
        print(f"  Succeeded: {job.stats.succeeded_request_count}")
        print(f"  Failed: {job.stats.failed_request_count}")

    return job


def parse_batch_results(job: Dict, questions: List[Dict]) -> List[tuple]:
    """
    Parse batch job results and match with original questions.

    For inline requests, results come back in the same order as requests.

    Args:
        job: Completed batch job
        questions: Original questions list

    Returns:
        List of (question_idx, llm_answer) tuples
    """
    results = []

    # Responses are in job.dest.inlined_responses for inline requests
    inlined_responses = None
    if hasattr(job, 'dest') and hasattr(job.dest, 'inlined_responses'):
        inlined_responses = job.dest.inlined_responses
    elif hasattr(job, 'inlined_responses'):
        inlined_responses = job.inlined_responses

    if not inlined_responses:
        print("Warning: No inlined responses found in batch job")
        return results

    # For inline requests, responses are in the same order as requests
    for idx, response_obj in enumerate(inlined_responses):
        # Extract answer text
        if response_obj.response and response_obj.response.candidates:
            answer = response_obj.response.candidates[0].content.parts[0].text.strip()
            results.append((idx, answer))
        else:
            # Request failed or had no response
            results.append((idx, "ERROR: No response from model"))

    return results


def evaluate_with_batch_api(data_path: str,
                            model: str,
                            output_path: str,
                            thinking_budget: int,
                            limit: int = None,
                            poll_interval: int = 30) -> Dict:
    """
    Evaluate dataset using Batch API (async, 50% cheaper).

    Args:
        data_path: Path to questions JSONL
        model: Gemini model name
        output_path: Where to save results
        thinking_budget: Thinking budget per request
        limit: Optional question limit
        poll_interval: Seconds between status checks

    Returns:
        Evaluation results dict
    """
    print("="*80)
    print("GENEVAL EVALUATION (BATCH API MODE)")
    print("="*80)
    print("Note: Batch API is 50% cheaper but async (may take minutes/hours)")

    # Initialize client
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found. Please set it in your .env file.")

    client = genai.Client(api_key=api_key)

    # Load questions
    print(f"\n[1/5] Loading questions from {data_path}...")
    questions = load_questions(data_path)
    if limit:
        questions = questions[:limit]
        print(f"✓ Loaded {len(questions)} questions (limited to {limit})")
    else:
        print(f"✓ Loaded {len(questions)} questions")

    # Prepare batch requests
    print(f"\n[2/5] Preparing batch requests...")
    print(f"  Model: {model}")
    print(f"  Thinking budget: {thinking_budget}")

    batch_requests = prepare_batch_requests(questions, thinking_budget)
    print(f"✓ Prepared {len(batch_requests)} batch requests")

    # Submit batch job
    print(f"\n[3/5] Submitting batch job...")
    try:
        batch_job = client.batches.create(
            model=f"models/{model}" if not model.startswith('models/') else model,
            src=batch_requests,
            config={'display_name': 'geneval-batch-job'}
        )
        print(f"✓ Batch job created: {batch_job.name}")
    except Exception as e:
        print(f"✗ Failed to create batch job: {e}")
        raise

    # Poll for completion
    print(f"\n[4/5] Waiting for batch completion...")
    final_job = poll_batch_completion(client, batch_job.name, poll_interval)

    if final_job.state.name == 'JOB_STATE_FAILED':
        raise Exception(f"Batch job failed: {final_job.error}")
    elif final_job.state.name != 'JOB_STATE_SUCCEEDED':
        raise Exception(f"Batch job ended with state: {final_job.state.name}")

    # Parse results
    print(f"\n[5/5] Processing results with LLM-as-judge...")
    print("="*80)

    batch_results = parse_batch_results(final_job, questions)

    # Create mapping of index -> answer
    llm_answers = {idx: answer for idx, answer in batch_results}

    # Batch compare all answers at once (more efficient)
    print("  Submitting batch comparison for all answers...")
    comparison_results = batch_compare_answers(client, questions, llm_answers, model)

    # Now process results
    results = []
    correct_count = 0
    task_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
    difficulty_stats = defaultdict(lambda: {'total': 0, 'correct': 0})

    for i, q in enumerate(questions):
        print(f"\n[Question {i+1}/{len(questions)}] ({q['difficulty'].upper()}) - {q['task_type']}")
        print(f"Q: {q['question'][:100]}..." if len(q['question']) > 100 else f"Q: {q['question']}")

        # Get LLM answer from batch results
        llm_answer = llm_answers.get(i, "ERROR: No answer found in batch results")
        print(f"LLM: {llm_answer[:100]}..." if len(llm_answer) > 100 else f"LLM: {llm_answer}")
        print(f"Expected: {q['answer'][:100]}..." if len(q['answer']) > 100 else f"Expected: {q['answer']}")

        # Get comparison result from batch (already computed)
        comparison = comparison_results[i]
        is_correct = comparison['is_correct']

        if is_correct:
            correct_count += 1
            print(f"✓ CORRECT")
        else:
            print(f"✗ INCORRECT")

        print(f"Reason: {comparison['explanation']}")

        # Update statistics
        task_stats[q['task_name']]['total'] += 1
        task_stats[q['task_name']]['correct'] += is_correct

        difficulty_stats[q['difficulty']]['total'] += 1
        difficulty_stats[q['difficulty']]['correct'] += is_correct

        # Store result
        results.append({
            'question': q['question'],
            'task_type': q['task_type'],
            'task_name': q['task_name'],
            'difficulty': q['difficulty'],
            'true_answer': q['answer'],
            'llm_answer': llm_answer,
            'is_correct': is_correct,
            'explanation': comparison['explanation'],
            'metadata': q.get('metadata', {})
        })

    # Calculate statistics
    print("\n" + "="*80)
    print("[6/6] Computing statistics...")

    accuracy = correct_count / len(questions) if questions else 0

    # Accuracy by task
    accuracy_by_task = {}
    for task_name, stats in task_stats.items():
        accuracy_by_task[task_name] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0

    # Accuracy by difficulty
    accuracy_by_difficulty = {}
    for difficulty, stats in difficulty_stats.items():
        accuracy_by_difficulty[difficulty] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0

    # Prepare results with batch-specific metadata
    eval_results = {
        'api_mode': 'batch',
        'batch_job_name': final_job.name,
        'batch_processing_time_seconds': time.time() - start_time if 'start_time' in locals() else None,
        'batch_stats': {
            'total_request_count': final_job.stats.total_request_count if hasattr(final_job, 'stats') else None,
            'succeeded_request_count': final_job.stats.succeeded_request_count if hasattr(final_job, 'stats') else None,
            'failed_request_count': final_job.stats.failed_request_count if hasattr(final_job, 'stats') else None,
        },
        'model': model,
        'thinking_budget': thinking_budget,
        'total_questions': len(questions),
        'correct': correct_count,
        'incorrect': len(questions) - correct_count,
        'overall_accuracy': accuracy,
        'accuracy_by_task': accuracy_by_task,
        'accuracy_by_difficulty': accuracy_by_difficulty,
        'task_statistics': {
            task: {'total': stats['total'], 'correct': stats['correct'], 'accuracy': accuracy_by_task[task]}
            for task, stats in task_stats.items()
        },
        'difficulty_statistics': {
            diff: {'total': stats['total'], 'correct': stats['correct'], 'accuracy': accuracy_by_difficulty[diff]}
            for diff, stats in difficulty_stats.items()
        },
        'results': results
    }

    # Save results
    print(f"✓ Saving results to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)

    # Print summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY (BATCH API)")
    print("="*80)
    print(f"Batch Job: {final_job.name}")
    print(f"Model: {model}")
    print(f"Thinking Budget: {thinking_budget} tokens")
    print(f"Total Questions: {len(questions)}")
    print(f"Correct: {correct_count}")
    print(f"Incorrect: {len(questions) - correct_count}")
    print(f"Overall Accuracy: {accuracy:.1%}")

    print(f"\nAccuracy by Task:")
    for task_name in sorted(accuracy_by_task.keys()):
        acc = accuracy_by_task[task_name]
        total = task_stats[task_name]['total']
        correct = task_stats[task_name]['correct']
        print(f"  {task_name:.<50} {acc:>6.1%}  ({correct}/{total})")

    print(f"\nAccuracy by Difficulty:")
    for difficulty in ['easy', 'medium', 'hard']:
        if difficulty in accuracy_by_difficulty:
            acc = accuracy_by_difficulty[difficulty]
            total = difficulty_stats[difficulty]['total']
            correct = difficulty_stats[difficulty]['correct']
            print(f"  {difficulty.capitalize():.<50} {acc:>6.1%}  ({correct}/{total})")

    print("\n" + "="*80)
    print(f"✓ Evaluation complete! Results saved to {output_path}")
    print("="*80)

    return eval_results


def compare_answers(client: genai.Client,
                   question: str,
                   llm_answer: str,
                   true_answer: str,
                   model: str) -> Dict:
    """
    Use LLM to compare if the LLM's answer matches the true answer.
    Returns dict with 'is_correct' boolean and 'explanation' string.
    """
    comparison_prompt = f"""You are evaluating whether an AI correctly answered a question.

Question: {question}
AI's Answer: {llm_answer}
Correct Answer: {true_answer}

IMPORTANT: Focus ONLY on whether the AI answered the QUESTION correctly, not on minor details in its reasoning.

Guidelines:
- For "Who" questions: Did the AI identify the correct person? (Ignore date formatting, titles)
- For comparison questions ("who was older/longer"): Did the AI pick the right person/option? (Ignore if exact ages are off by 1-2 years)
- For "How many" questions: Did the AI give the exact number? (Must be exact for counting)
- For "True/False" questions: Did the AI give the correct boolean answer?
- For numerical questions with units: Accept if within ±1 of the correct value due to date calculation methods

Respond with ONLY a JSON object in this exact format:
{{
  "is_correct": true or false,
  "explanation": "Brief explanation focusing on whether the core question was answered correctly"
}}"""

    try:
        response = client.models.generate_content(
            model=model,
            contents=comparison_prompt,
        )
        response_text = response.text.strip()

        # Extract JSON from markdown code blocks if present
        if '```json' in response_text:
            response_text = response_text.split('```json')[1].split('```')[0].strip()
        elif '```' in response_text:
            response_text = response_text.split('```')[1].split('```')[0].strip()

        result = json.loads(response_text)
        return result
    except Exception as e:
        return {
            "is_correct": False,
            "explanation": f"Failed to parse comparison result: {str(e)}"
        }


def batch_compare_answers(client: genai.Client,
                          questions: List[Dict],
                          llm_answers: Dict[int, str],
                          model: str) -> List[Dict]:
    """
    Compare multiple answers using batch API (more efficient).

    Args:
        client: Gemini client
        questions: List of question dicts
        llm_answers: Dict mapping question index to LLM answer
        model: Model name

    Returns:
        List of comparison results (in same order as questions)
    """
    print("  Preparing comparison batch...")

    # Prepare batch comparison requests
    comparison_requests = []
    for i, q in enumerate(questions):
        llm_answer = llm_answers.get(i, "ERROR: No answer found in batch results")

        comparison_prompt = f"""You are evaluating whether an AI correctly answered a question.

Question: {q['question']}
AI's Answer: {llm_answer}
Correct Answer: {q['answer']}

IMPORTANT: Focus ONLY on whether the AI answered the QUESTION correctly, not on minor details in its reasoning.

Guidelines:
- For "Who" questions: Did the AI identify the correct person? (Ignore date formatting, titles)
- For comparison questions ("who was older/longer"): Did the AI pick the right person/option? (Ignore if exact ages are off by 1-2 years)
- For "How many" questions: Did the AI give the exact number? (Must be exact for counting)
- For "True/False" questions: Did the AI give the correct boolean answer?
- For numerical questions with units: Accept if within ±1 of the correct value due to date calculation methods

Respond with ONLY a JSON object in this exact format:
{{
  "is_correct": true or false,
  "explanation": "Brief explanation focusing on whether the core question was answered correctly"
}}"""

        comparison_requests.append({
            'contents': [{
                'parts': [{'text': comparison_prompt}],
                'role': 'user'
            }]
        })

    print(f"  Submitting comparison batch ({len(comparison_requests)} comparisons)...")

    # Submit batch job for comparisons
    comparison_job = client.batches.create(
        model=f"models/{model}" if not model.startswith('models/') else model,
        src=comparison_requests,
        config={'display_name': 'geneval-comparison-batch'}
    )

    print(f"  Comparison batch job: {comparison_job.name}")
    print(f"  Polling for completion...")

    # Poll for completion (shorter interval since comparisons are faster)
    comparison_job = poll_batch_completion(client, comparison_job.name, poll_interval=10)

    # Parse comparison results
    comparison_results = []

    inlined_responses = None
    if hasattr(comparison_job, 'dest') and hasattr(comparison_job.dest, 'inlined_responses'):
        inlined_responses = comparison_job.dest.inlined_responses
    elif hasattr(comparison_job, 'inlined_responses'):
        inlined_responses = comparison_job.inlined_responses

    if inlined_responses:
        for response_obj in inlined_responses:
            if response_obj.response and response_obj.response.candidates:
                response_text = response_obj.response.candidates[0].content.parts[0].text.strip()

                # Parse JSON from response
                try:
                    # Extract JSON from markdown if present
                    if '```json' in response_text:
                        response_text = response_text.split('```json')[1].split('```')[0].strip()
                    elif '```' in response_text:
                        response_text = response_text.split('```')[1].split('```')[0].strip()

                    result = json.loads(response_text)
                    comparison_results.append(result)
                except Exception as e:
                    comparison_results.append({
                        "is_correct": False,
                        "explanation": f"Failed to parse comparison: {str(e)}"
                    })
            else:
                comparison_results.append({
                    "is_correct": False,
                    "explanation": "No response from comparison model"
                })

    return comparison_results


def evaluate_dataset(data_path: str = 'data/examples.jsonl',
                    model: str = "gemini-2.5-flash",
                    output_path: str = 'data/evaluation_results.json',
                    limit: int = None) -> Dict:
    """
    Run complete evaluation on the dataset.

    Args:
        data_path: Path to JSONL file with questions
        model: Gemini model to use
        output_path: Where to save results
        limit: Optional limit on number of questions to evaluate (for testing)

    Returns:
        Dict with evaluation results and statistics
    """
    print("="*80)
    print("GENEVAL EVALUATION")
    print("="*80)

    # Initialize Gemini client
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found. Please set it in your .env file.")

    client = genai.Client(api_key=api_key)

    # Get thinking budget for this model
    thinking_budget = get_thinking_budget(model)
    print(f"Using thinking budget: {thinking_budget} tokens for {model}")

    # Load questions
    print(f"\n[1/3] Loading questions from {data_path}...")
    questions = load_questions(data_path)
    if limit:
        questions = questions[:limit]
        print(f"✓ Loaded {len(questions)} questions (limited to {limit} for testing)")
    else:
        print(f"✓ Loaded {len(questions)} questions")

    # Run evaluation
    print(f"\n[2/3] Evaluating with {model}...")
    print("="*80)

    results = []
    correct_count = 0
    task_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
    difficulty_stats = defaultdict(lambda: {'total': 0, 'correct': 0})

    for i, q in enumerate(questions, 1):
        print(f"\n[Question {i}/{len(questions)}] ({q['difficulty'].upper()}) - {q['task_type']}")
        print(f"Q: {q['question'][:100]}..." if len(q['question']) > 100 else f"Q: {q['question']}")

        # Get LLM's answer
        llm_answer = get_llm_answer(client, q['question'], model, thinking_budget)
        print(f"LLM: {llm_answer[:100]}..." if len(llm_answer) > 100 else f"LLM: {llm_answer}")
        print(f"Expected: {q['answer'][:100]}..." if len(q['answer']) > 100 else f"Expected: {q['answer']}")

        # Compare answers
        comparison = compare_answers(client, q['question'], llm_answer, q['answer'], model)
        is_correct = comparison['is_correct']

        if is_correct:
            correct_count += 1
            print(f"✓ CORRECT")
        else:
            print(f"✗ INCORRECT")

        print(f"Reason: {comparison['explanation']}")

        # Update statistics
        task_stats[q['task_name']]['total'] += 1
        task_stats[q['task_name']]['correct'] += is_correct

        difficulty_stats[q['difficulty']]['total'] += 1
        difficulty_stats[q['difficulty']]['correct'] += is_correct

        # Store result
        results.append({
            'question': q['question'],
            'task_type': q['task_type'],
            'task_name': q['task_name'],
            'difficulty': q['difficulty'],
            'true_answer': q['answer'],
            'llm_answer': llm_answer,
            'is_correct': is_correct,
            'explanation': comparison['explanation'],
            'metadata': q.get('metadata', {})
        })

        # Rate limiting disabled - using API key with higher quota
        # Uncomment below if you hit rate limits:
        # if i < len(questions):
        #     wait_time = 7
        #     print(f"  [Rate limiting: waiting {wait_time}s...]")
        #     time.sleep(wait_time)

    # Calculate final statistics
    print("\n" + "="*80)
    print("[3/3] Computing statistics...")

    accuracy = correct_count / len(questions) if questions else 0

    # Accuracy by task
    accuracy_by_task = {}
    for task_name, stats in task_stats.items():
        accuracy_by_task[task_name] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0

    # Accuracy by difficulty
    accuracy_by_difficulty = {}
    for difficulty, stats in difficulty_stats.items():
        accuracy_by_difficulty[difficulty] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0

    # Prepare results
    eval_results = {
        'api_mode': 'single',
        'model': model,
        'thinking_budget': thinking_budget,
        'total_questions': len(questions),
        'correct': correct_count,
        'incorrect': len(questions) - correct_count,
        'overall_accuracy': accuracy,
        'accuracy_by_task': accuracy_by_task,
        'accuracy_by_difficulty': accuracy_by_difficulty,
        'task_statistics': {
            task: {'total': stats['total'], 'correct': stats['correct'], 'accuracy': accuracy_by_task[task]}
            for task, stats in task_stats.items()
        },
        'difficulty_statistics': {
            diff: {'total': stats['total'], 'correct': stats['correct'], 'accuracy': accuracy_by_difficulty[diff]}
            for diff, stats in difficulty_stats.items()
        },
        'results': results
    }

    # Save results
    print(f"✓ Saving results to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)

    # Print summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"Model: {model}")
    print(f"Thinking Budget: {thinking_budget} tokens")
    print(f"Total Questions: {len(questions)}")
    print(f"Correct: {correct_count}")
    print(f"Incorrect: {len(questions) - correct_count}")
    print(f"Overall Accuracy: {accuracy:.1%}")

    print(f"\nAccuracy by Task:")
    for task_name in sorted(accuracy_by_task.keys()):
        acc = accuracy_by_task[task_name]
        total = task_stats[task_name]['total']
        correct = task_stats[task_name]['correct']
        print(f"  {task_name:.<50} {acc:>6.1%}  ({correct}/{total})")

    print(f"\nAccuracy by Difficulty:")
    for difficulty in ['easy', 'medium', 'hard']:
        if difficulty in accuracy_by_difficulty:
            acc = accuracy_by_difficulty[difficulty]
            total = difficulty_stats[difficulty]['total']
            correct = difficulty_stats[difficulty]['correct']
            print(f"  {difficulty.capitalize():.<50} {acc:>6.1%}  ({correct}/{total})")

    print("\n" + "="*80)
    print(f"✓ Evaluation complete! Results saved to {output_path}")
    print("="*80)

    return eval_results


def main():
    """Main entry point for command-line usage"""
    parser = argparse.ArgumentParser(
        description='Run GENEVAL evaluation on generated dataset'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='data/examples.jsonl',
        help='Path to dataset JSONL file (default: data/examples.jsonl)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='gemini-2.5-flash',
        help='Gemini model to use (default: gemini-2.5-flash, supported: gemini-2.5-flash, gemini-2.5-pro)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/evaluation_results.json',
        help='Output file for results (default: data/evaluation_results.json)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of questions (for testing)'
    )

    args = parser.parse_args()

    # Get model from environment variable or command line argument
    # Priority: env var > command line > default
    model = os.getenv('GEMINI_MODEL', args.model)

    # Check if batch API mode is enabled
    use_batch_api = os.getenv('USE_BATCH_API', 'false').lower() == 'true'

    print(f"Using model: {model}")
    if os.getenv('GEMINI_MODEL'):
        print(f"  (from GEMINI_MODEL environment variable)")

    print(f"API Mode: {'Batch API (async, 50% cheaper)' if use_batch_api else 'Single API (real-time)'}")
    if use_batch_api:
        print("  Note: Batch mode may take minutes/hours for completion")

    # Run evaluation with appropriate method
    if use_batch_api:
        # Get thinking budget
        thinking_budget = get_thinking_budget(model)

        results = evaluate_with_batch_api(
            data_path=args.data,
            model=model,
            output_path=args.output,
            thinking_budget=thinking_budget,
            limit=args.limit,
            poll_interval=30  # Poll every 30 seconds
        )
    else:
        results = evaluate_dataset(
            data_path=args.data,
            model=model,
            output_path=args.output,
            limit=args.limit
        )

    print(f"\nFinal Accuracy: {results['overall_accuracy']:.1%}")


if __name__ == '__main__':
    main()
