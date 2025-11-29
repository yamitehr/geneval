# GENEVAL: Genealogical Reasoning Evaluation Benchmark

A comprehensive benchmark for testing Large Language Models on structured knowledge reasoning through genealogical questions about the British Royal Family.

## 📋 Overview

**GENEVAL** challenges LLMs across **7 diverse reasoning types**, testing capabilities that go beyond simple fact retrieval:

1. **Multi-Hop Reasoning** - Compositional reasoning through chained parent/spouse relationships (1-6 hops)
2. **Temporal Reasoning** - Date arithmetic, chronological ordering, and temporal logic
3. **Comparative Lifespan** - Numerical comparison with historical context
4. **Negative Reasoning** - Handling negation and false premises (known LLM weakness)
5. **Sibling Inference** - Deriving implicit relationships from explicit data
6. **Constraint Satisfaction** - Finding entities matching multiple simultaneous constraints
7. **Adversarial Ambiguity** - Disambiguating people with identical names across dynasties

### Key Features

- **501 questions** across 7 task types
- **3 difficulty levels** per task (easy, medium, hard)
- **Rich metadata** for error analysis (task type, difficulty, reasoning requirements)
- **Automated generation** - scalable to 1000s of questions
- **Verifiable ground truth** - all answers deterministic and checkable

---

## 🚀 Quick Start

### Prerequisites

- Python >= 3.10

### Setup

1. **Clone and install dependencies:**
```bash
cd nlp_final_project_313600751
pip install -r requirements.txt  # installs google-genai, python-dotenv, rich
```

2. **Set up your Gemini API key:**
Create a `.env` file:
```bash
GOOGLE_API_KEY=your_api_key_here

# Optional: Specify which model to use
# Supported models: gemini-2.5-flash, gemini-2.5-pro
# Default: gemini-2.5-flash
GEMINI_MODEL=gemini-2.5-flash

# Optional: Thinking budget (number of thinking tokens)
# Special values:
#   -1 = Dynamic thinking (model decides when/how much to think) - DEFAULT
#    0 = Disable thinking (only for Flash models)
#
# Ranges per model:
#   gemini-2.5-flash: 0 to 24576
#   gemini-2.5-pro:   128 to 32768 (cannot disable thinking)
#
# Recommended for complex reasoning tasks: 8192-16384
THINKING_BUDGET=-1

# Batch API mode (optional)
# false = Single API calls (default, real-time, immediate results)
# true = Batch API (50% cheaper, async processing, results in minutes/hours)
USE_BATCH_API=true
```

### Usage

#### 1. Generate Dataset

Generate the complete dataset of 501 questions:

```bash
python generate_data.py
```

This will:
- Load British Royal Family data (314 people)
- Generate questions from 7 task types
- Save to `data/examples.jsonl`
- Print comprehensive statistics

**Options:**
```bash
python generate_data.py --count 200 --output data/custom.jsonl
```

#### 2. Run Evaluation

Evaluate an LLM on the generated dataset:

```bash
python run_eval.py
```

This will:
- Load questions from `data/examples.jsonl`
- Query Gemini model for each question
- Use LLM-as-judge for answer comparison
- Compute accuracy by task and difficulty
- Save results to `data/evaluation_results.json`

**Options:**
```bash
python run_eval.py --model gemini-2.5-pro --limit 50
```

**Supported Models:**
- `gemini-2.5-flash` (default) - Faster, thinking budget range: 0-24,576 tokens
- `gemini-2.5-pro` - More capable, thinking budget range: 128-32,768 tokens

**Thinking Budget:**
- `-1` (default): Dynamic thinking - model decides when and how much to think
- `0`: Disable thinking (Flash only, not recommended for complex reasoning)
- Custom value: Specify explicit token budget (enforces model-specific ranges)

Configure via `THINKING_BUDGET` environment variable. For complex reasoning tasks, recommended values are 8,192-16,384 tokens.

#### 3. Analyze Results

After running an evaluation, analyze the results with detailed breakdowns:

```bash
python analyze_results.py --results data/evaluation_results.json --csv british_royal_family_FINAL.csv
```

This will show accuracy statistics segmented by:
- **Task Type** - Performance on each of the 7 reasoning tasks
- **Difficulty** - Easy, medium, hard breakdown
- **Royal House** - Tudor, Stuart, Windsor, etc.
- **Time Period** - Medieval, Early Modern, Modern
- **Century** - 9th through 21st century
- **People Involved** - Number of entities in each question
- **Monarch Involvement** - Questions involving monarchs vs. non-monarchs
- **Gender Composition** - All male, all female, or mixed
- **Relationship Type** - Parent-child, sibling, spouse, etc.
- **Dynasty** - Single house vs. cross-house questions
- **Hop Count** - Compositional degradation curve (2-6 hops)

**Example Output:**

```
╭──────────────────────────────────────────────────────────────╮
│ Evaluation Results: gemini-2.5-pro | Overall Accuracy: 82.8% │
╰──────────────────────────────────────────────────────────────╯

                       Complete Analysis (501 questions)
╭──────────────────────┬─────────────────────────────────────┬───────┬─────────╮
│ Segmentation         │ Category                            │ Accu… │ Correc… │
├──────────────────────┼─────────────────────────────────────┼───────┼─────────┤
│ TASK TYPE            │                                     │       │         │
├──────────────────────┼─────────────────────────────────────┼───────┼─────────┤
│ Task Type            │ Temporal Reasoning                  │ 98.6% │  71/72  │
│ Task Type            │ Adversarial Ambiguity               │ 98.4% │  62/63  │
│ Task Type            │ Sibling Inference                   │ 90.3% │  65/72  │
│ Task Type            │ Negative Reasoning                  │ 81.9% │  59/72  │
│ Task Type            │ Comparative Lifespan Reasoning      │ 78.3% │  47/60  │
│ ...                  │ ...                                 │ ...   │  ...    │
├──────────────────────┼─────────────────────────────────────┼───────┼─────────┤
│ DIFFICULTY           │                                     │       │         │
├──────────────────────┼─────────────────────────────────────┼───────┼─────────┤
│ Difficulty           │ Easy                                │ 89.0% │ 113/127 │
│ Difficulty           │ Medium                              │ 85.4% │ 182/213 │
│ Difficulty           │ Hard                                │ 74.5% │ 120/161 │
├──────────────────────┼─────────────────────────────────────┼───────┼─────────┤
│ ROYAL HOUSE          │                                     │       │         │
├──────────────────────┼─────────────────────────────────────┼───────┼─────────┤
│ Royal House          │ House of Saxe-Coburg and Gotha      │ 100.% │   3/3   │
│ Royal House          │ House of Windsor                    │ 91.4% │  32/35  │
│ Royal House          │ House of Tudor                      │ 88.5% │  85/96  │
│ Royal House          │ House of Hanover                    │ 87.5% │  14/16  │
│ Royal House          │ House of Plantagenet                │ 86.9% │ 186/214 │
│ ...                  │ ...                                 │ ...   │  ...    │
├──────────────────────┼─────────────────────────────────────┼───────┼─────────┤
│ TIME PERIOD          │                                     │       │         │
├──────────────────────┼─────────────────────────────────────┼───────┼─────────┤
│ Time Period          │ Modern                              │ 85.7% │  60/70  │
│ Time Period          │ Medieval                            │ 83.6% │ 317/379 │
│ Time Period          │ Early Modern                        │ 81.5% │ 154/189 │
├──────────────────────┼─────────────────────────────────────┼───────┼─────────┤
│ CENTURY (TOP 5)      │                                     │       │         │
├──────────────────────┼─────────────────────────────────────┼───────┼─────────┤
│ Century              │ 9th century                         │ 89.5% │  17/19  │
│ Century              │ 13th century                        │ 87.2% │  82/94  │
│ Century              │ 10th century                        │ 85.4% │  35/41  │
│ Century              │ 12th century                        │ 82.9% │  63/76  │
│ Century              │ 11th century                        │ 73.9% │  51/69  │
│ ...                  │ ...                                 │ ...   │  ...    │
├──────────────────────┼─────────────────────────────────────┼───────┼─────────┤
│ PEOPLE INVOLVED      │                                     │       │         │
├──────────────────────┼─────────────────────────────────────┼───────┼─────────┤
│ People Involved      │ 5 people                            │ 91.7% │  11/12  │
│ People Involved      │ 2 people                            │ 87.0% │ 87/100  │
│ People Involved      │ 1 person                            │ 83.1% │ 212/255 │
│ People Involved      │ 3 people                            │ 81.9% │  68/83  │
│ People Involved      │ 4 people                            │ 81.8% │  27/33  │
│ ...                  │ ...                                 │ ...   │  ...    │
├──────────────────────┼─────────────────────────────────────┼───────┼─────────┤
│ MONARCH              │                                     │       │         │
├──────────────────────┼─────────────────────────────────────┼───────┼─────────┤
│ Monarch              │ Has monarch                         │ 85.6% │  83/97  │
│ Monarch              │ No monarchs                         │ 85.4% │ 229/268 │
│ Monarch              │ All monarchs                        │ 75.7% │ 103/136 │
├──────────────────────┼─────────────────────────────────────┼───────┼─────────┤
│ GENDER               │                                     │       │         │
├──────────────────────┼─────────────────────────────────────┼───────┼─────────┤
│ Gender               │ All females                         │ 86.2% │ 94/109  │
│ Gender               │ All males                           │ 83.9% │ 209/249 │
│ Gender               │ Mixed                               │ 80.3% │ 102/127 │
│ ...                  │ ...                                 │ ...   │  ...    │
├──────────────────────┼─────────────────────────────────────┼───────┼─────────┤
│ RELATIONSHIP         │                                     │       │         │
├──────────────────────┼─────────────────────────────────────┼───────┼─────────┤
│ Relationship         │ Disambiguation                      │ 98.4% │  61/62  │
│ Relationship         │ Sibling                             │ 91.2% │  52/57  │
│ Relationship         │ Temporal                            │ 89.5% │ 119/133 │
│ Relationship         │ Parent-Child                        │ 83.0% │  39/47  │
│ Relationship         │ Negative                            │ 82.5% │  33/40  │
│ ...                  │ ...                                 │ ...   │  ...    │
├──────────────────────┼─────────────────────────────────────┼───────┼─────────┤
│ DYNASTY              │                                     │       │         │
├──────────────────────┼─────────────────────────────────────┼───────┼─────────┤
│ Dynasty              │ Single house                        │ 83.2% │ 263/316 │
│ Dynasty              │ Cross-house                         │ 82.2% │ 152/185 │
├──────────────────────┼─────────────────────────────────────┼───────┼─────────┤
│ HOP COUNT (Compositional Degradation)                      │       │         │
├──────────────────────┼─────────────────────────────────────┼───────┼─────────┤
│ Hop Count            │ 2 hops                              │ 81.8% │  27/33  │
│ Hop Count            │ 3 hops                              │ 64.7% │  11/17  │
│ Hop Count            │ 4 hops                              │ 62.5% │  10/16  │
│ Hop Count            │ 5 hops                              │ 58.8% │  10/17  │
│ Hop Count            │ 6 hops                              │ 43.8% │  7/16   │
╰──────────────────────┴─────────────────────────────────────┴───────┴─────────╯

╭──────────────────────╮
│ ✓ Analysis Complete! │
╰──────────────────────╯
```

The hop count analysis clearly demonstrates the **compositional degradation** phenomenon: accuracy drops from 81.8% at 2 hops down to 43.8% at 6 hops, confirming that LLMs struggle with multi-step reasoning chains.

---

### Batch API Mode (Cost Optimization)

The evaluation supports two API modes for comparing cost vs. latency:

#### Single API Mode (Default)
- **When to use:** Quick tests, real-time feedback needed, < 50 questions
- **Speed:** Immediate results (seconds per question)
- **Cost:** Standard API pricing
- **Setup:** `USE_BATCH_API=false` (default)

#### Batch API Mode
- **When to use:** Large-scale evaluations (100+ questions), cost-sensitive workloads
- **Speed:** Async processing, typically completes within minutes to hours
- **Cost:** **50% cheaper** than single API calls
- **Setup:** `USE_BATCH_API=true` in `.env`

**Example:**
```bash
# Enable batch mode in .env
USE_BATCH_API=true

# Run evaluation
python run_eval.py

# The script will:
# 1. Submit all questions as a batch job
# 2. Poll every 30 seconds for completion
# 3. Process results once ready
# 4. Save with batch metadata for comparison
```

**Output includes batch-specific metadata:**
- `api_mode`: "single" or "batch"
- `batch_job_name`: Job identifier
- `batch_processing_time_seconds`: Total time
- `batch_stats`: Success/failure counts

---

## 📁 Project Structure

```
nlp_final_project_313600751/
├── generate_data.py         # Main script to generate dataset
├── run_eval.py              # Main script to run evaluation
├── analyze_results.py       # Detailed results analysis with segmentations
├── evaluation.py            # Evaluation functions (used by run_eval.py)
├── README.md                # This file
│
├── data/
│   ├── examples.jsonl       # Generated dataset
│   └── evaluation_results.json  # Evaluation results
│
├── Data & Infrastructure:
│   ├── british_royal_family_FINAL.csv  # Source data (314 British Royal Family Members)
│   └── data_loader.py       # Enhanced data loader with computed relationships
│
└── Task Generators (modular design):
    ├── task_base.py                      # Base classes for all tasks
    ├── task_1_multihop.py                # Multi-hop reasoning
    ├── task_2_temporal.py                # Temporal reasoning
    ├── task_3_lifespan.py                # Comparative lifespan
    ├── task_4_negative.py                # Negative reasoning
    ├── task_5_siblings.py                # Sibling inference
    ├── task_6_constraint_satisfaction.py # Constraint satisfaction
    └── task_7_adversarial_ambiguity.py   # Adversarial ambiguity
```

---

## 📊 Dataset Structure

Each question in `data/examples.jsonl` follows this format:

```json
{
  "question": "Who is the father of the mother of Edward III of England (1312–1377)?",
  "answer": "Henry I of England (1068–1135)",
  "task_type": "multi_hop",
  "task_name": "Multi-Hop Relationship Traversal",
  "difficulty": "medium",
  "metadata": {
    "hops": 3,
    "start_person": "Edward III of England (1312–1377)",
    "path": ["mother", "father"],
    "requires_compositional_reasoning": true
  }
}
```

### Dataset Statistics

- **Total questions:** 501
- **Task distribution:**
  - Multi-Hop Reasoning: 99 questions (20%)
  - Temporal Reasoning: 72 questions (14%)
  - Negative Reasoning: 72 questions (14%)
  - Sibling Inference: 72 questions (14%)
  - Constraint Satisfaction: 63 questions (13%)
  - Adversarial Ambiguity: 63 questions (13%)
  - Comparative Lifespan: 60 questions (12%)

- **Difficulty distribution:**
  - Easy: 33%
  - Medium: 33%
  - Hard: ~33%

---

## 🧪 Task Descriptions

### Task 1: Multi-Hop Reasoning
**Tests:** Compositional reasoning through relationship chains
**Examples:**
- Easy (1-2 hops): "Who is the father of X?"
- Medium (3-4 hops): "Who is the father of the mother of X?"
- Hard (5-6 hops): "Who is the mother of the father of the spouse of the mother of X?"

### Task 2: Temporal Reasoning
**Tests:** Date arithmetic, chronological logic
**Examples:**
- "Who was born first, X or Y?"
- "Did X die before Y was born?"
- "How old was X when Y was born?"

### Task 3: Comparative Lifespan
**Tests:** Numerical comparison with context
**Examples:**
- "Who lived longer, X or Y?"
- "Who was older when crowned, X or Y?"
- "By how many years did X outlive Y?"

### Task 4: Negative Reasoning
**Tests:** Handling negation (known LLM weakness)
**Examples:**
- "Who among [A, B, C] is NOT a child of X?"
- "True or False: X is the grandparent of Y"
- "Which of these people did NOT live in the 1800s?"

### Task 5: Sibling Inference
**Tests:** Deriving implicit relationships
**Examples:**
- "Name a sibling of X"
- "Did X and Y have the same mother AND father?"
- "How many full siblings did X have?"

### Task 6: Constraint Satisfaction
**Tests:** Finding entities matching multiple criteria
**Examples:**
- "Name a female monarch from the House of Tudor who reigned in the 16th century"
- "Who was a male child of X born after 1500?"

### Task 7: Adversarial Ambiguity
**Tests:** Disambiguating identical names across dynasties
**Examples:**
- "Which Edward was the son of Queen Victoria?" (vs. medieval Edwards)
- "Which Henry married Anne Boleyn?" (vs. other Henrys)

---

## 🔬 Why This Benchmark Matters

### Tests Multiple Reasoning Types
Unlike benchmarks that focus on single capabilities, GENEVAL tests:
- **Compositional reasoning** (chaining 4-6 relationship hops)
- **Temporal logic** (date arithmetic, chronology)
- **Numerical reasoning** (lifespan comparison)
- **Negative reasoning** (handling NOT, false premises)
- **Inference** (deriving implicit sibling relationships)
- **Constraint satisfaction** (multiple simultaneous criteria)
- **Disambiguation** (adversarial same-name confusion)

### Addresses Known LLM Weaknesses
- **Negation handling** - LLMs struggle with NOT questions
- **Multi-hop reasoning** - Accuracy degrades from ~95% (2 hops) to ~60% (6 hops)
- **Implicit relationships** - Must infer siblings from parents
- **Compositional generalization** - Combining multiple reasoning steps
- **Working memory limits** - Constraint satisfaction with 3-4 conditions

### Scalable & Extensible
- **Automated generation** - can create 1000s of questions
- **Verifiable ground truth** - all answers computed, not manually labeled
- **Modular design** - easy to add new task types
- **Rich metadata** - enables deep error analysis

---

## 📈 Expected Results

### Hypotheses
1. **H1:** Accuracy degrades exponentially with hop count (multi-hop tasks)
2. **H2:** LLMs perform worse on negative reasoning than positive
3. **H3:** Implicit relationships (siblings) harder than explicit (parents)
4. **H4:** Temporal reasoning worse than structural reasoning

### Evaluation Metrics
- **Overall accuracy** - percentage of correct answers
- **Per-task accuracy** - identifies reasoning type weaknesses
- **Per-difficulty accuracy** - measures scaling with complexity
- **Error type analysis** - categorizes failure modes

---

## 🛠️ Extending the Benchmark

### Adding New Tasks

1. Create a new task generator inheriting from `TaskGenerator`:

```python
# task_8_custom.py
from task_base import TaskGenerator

class CustomTask(TaskGenerator):
    def __init__(self, data_loader):
        super().__init__(data_loader, "Custom Task", "custom")

    def generate_questions(self, count, difficulty='mixed'):
        # Your generation logic
        pass
```

2. Import in `generate_data.py`:

```python
from task_8_custom import CustomTask

tasks = [
    # ... existing tasks
    (CustomTask(loader), 20),
]
```

### Modifying Question Counts

Edit the task allocation in `generate_data.py`:

```python
tasks = [
    (MultiHopTraversalTask(loader), 50),  # Increase multi-hop
    (TemporalReasoningTask(loader), 20),  # Decrease temporal
    # ...
]
```

---

## 📚 Data Source

- **Source:** Wikidata dump of British Royal Family
- **Size:** 314 people spanning 1000+ years (849 CE - 2025 CE)
- **Fields:** Name, dates, parents, spouse, reign periods, house, gender
- **Preprocessing:** Computed derived relationships (siblings, generation distances, lifespans)

### Data Quality
- **People:** 314 (fetched from Wikidata)
- **People with lifespans:** ~280 (89%)
- **Monarchs with reign data:** 50+
- **Sibling pairs:** Inferred from shared parents

---

## 🎯 Key Findings

### The Multi-Hop Reasoning Gap

Our evaluation with Gemini 2.5 reveals a fundamental gap between knowledge retrieval and compositional reasoning:

| Task Type | Accuracy |
|-----------|----------|
| Adversarial Ambiguity | **100%** |
| Temporal Reasoning | **100%** |
| Negative Reasoning | **100%** |
| Comparative Lifespan | 95% |
| Sibling Inference | 92.5% |
| Constraint Satisfaction | 82.5% |
| **Multi-Hop Reasoning** | **68.8%** |

**Key insight:** While the model achieves 100% accuracy on tasks proving it *knows* the facts, accuracy drops to 68.8% on multi-hop reasoning tasks. This reveals that LLMs excel at factual recall but struggle with compositional reasoning over known facts.

---

## 🐛 Troubleshooting

### API Rate Limits
If you hit Gemini API rate limits, adjust the wait time in `run_eval.py`:
```python
wait_time = 10  # Increase from 7 to 10 seconds
```

### Memory Issues
For large datasets, process in batches:
```bash
python run_eval.py --limit 50  # Evaluate first 50 questions
```

### Missing Dependencies
```bash
pip install -r requirements.txt
```

---

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact yamitehrlich@gmail.com.

---

**Last Updated:** November 2025
**Version:** 1.0
**License:** MIT
