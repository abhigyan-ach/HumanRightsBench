# Human Rights Benchmark Evaluation Tool

A comprehensive evaluation framework for assessing Large Language Models (LLMs) on human rights scenarios. This tool evaluates various LLMs across multiple providers (OpenAI, Anthropic, Google) on their understanding and reasoning about human rights violations, obligations, and remedies.

## Overview

The evaluation tool processes scenarios from the Human Rights Benchmark dataset and evaluates LLM responses across different question types:
- **Multiple Choice Questions (MCQ)**: Single-choice questions with letter answers (A, B, C, etc.)
- **Ranking Questions (R)**: Questions requiring ordered responses (e.g., "A,B,C")
- **Short Answer Questions (P)**: Open-ended questions requiring detailed responses

## Features

- ✅ Multi-provider support (OpenAI, Anthropic, Google, Qwen)
- ✅ Structured output using Pydantic models
- ✅ Automatic scoring for MCQ and ranking questions
- ✅ JSON response parsing
- ✅ Comprehensive evaluation metrics and reporting
- ✅ CSV export of results
- ✅ Dry-run mode for testing prompts
- ✅ Configurable temperature and token limits
- ✅ Progress tracking and error handling

## Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Required Files

- `scenarios.json` - The benchmark dataset containing human rights scenarios
- `model_choices.py` - LLM client wrapper for different providers
- `eval.py` - Main evaluation script

## Configuration

### API Keys

Set up your API keys for the providers you want to use:

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# Google
export GOOGLE_API_KEY="..."
```

Alternatively, pass API keys directly via command line using `--api_key`.

## Usage

### Basic Command

```bash
python eval.py --provider <provider> --model <model_name>
```

### Command Line Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--provider` | Yes | - | LLM provider: `openai`, `anthropic`, `google`, or `qwen` |
| `--model` | Yes | - | Model name (e.g., `gpt-4o`, `claude-3-5-sonnet-20241022`) |
| `--temperature` | No | 0.0 | Sampling temperature (0.0 = deterministic, 1.0 = creative) |
| `--max-tokens` | No | 1000 | Maximum tokens in response |
| `--api_key` | No | None | API key (overrides environment variables) |
| `--dry-run` | No | False | Print prompts without calling API (for testing) |
| `--limit` | No | None | Limit number of evaluations to run |

## Examples

### 1. Test with Dry Run (No API Calls)

```bash
python eval.py --provider openai --model gpt-4o --dry-run --limit 5
```

### 2. Evaluate with OpenAI GPT-4

```bash
python eval.py --provider openai --model gpt-4o --api_key "sk-..."
```

### 3. Evaluate with Anthropic Claude

```bash
python eval.py --provider anthropic --model claude-3-5-sonnet-20241022 --temperature 0.5
```

### 4. Evaluate with Google Gemini

```bash
python eval.py --provider google --model gemini-pro --limit 10
```

### 5. Custom Temperature and Token Limits

```bash
python eval.py --provider openai --model gpt-4o --temperature 0.7 --max-tokens 2000
```

## Output

### Console Output

The script displays real-time progress including:
- Current evaluation progress (e.g., "Progress: 5/184")
- Question ID and type
- Full prompt sent to the LLM
- LLM response (raw JSON or text)
- Correct answer
- Score (1.0 for correct, 0.0 for incorrect, -1 for manual evaluation)

### CSV Output

Results are saved to a CSV file named:
```
eval_results_{provider}_{model}.csv
```

#### CSV Columns:
- `scenario_id` - Unique scenario identifier
- `scenario_text` - Full scenario description
- `tags` - Scenario categorization tags
- `difficulty` - Difficulty level (easy, medium, hard)
- `subscenario_id` - Sub-scenario identifier
- `subscenario_text` - Specific sub-scenario description
- `question_id` - Question type (MCQ letters, R for ranking, P for short answer)
- `question_text` - The actual question
- `answer_choices` - Available answer options
- `correct_answer` - Ground truth answer
- `llm_response` - LLM's response (JSON or text)
- `score` - Evaluation score

### Summary Statistics

At the end of evaluation, the script prints:
- Average score (percentage)
- Total questions evaluated
- Number of correct answers
- Number of incorrect answers

Example:
```
================================================================================
Results saved to: eval_results_openai_gpt-4o.csv
Average Score: 76.32%
Total Questions: 152
Correct: 116
Incorrect: 36
================================================================================
```

## Question Types

### 1. Multiple Choice (Single Answer)

Questions with one correct answer from options A-H.

**Example:**
```
Question: Did the state violate any human rights obligations?
Options: A. Obligation to respect
         B. Obligation to protect
         C. No obligation violated
Correct Answer: B
```

### 2. Ranking Questions (R)

Questions requiring a specific order of answers.

**Example:**
```
Question: Rank these obligations from most to least violated
Correct Answer: A,B,C,D
```

### 3. Short Answer (P)

Open-ended questions requiring detailed responses (manual evaluation required).

**Example:**
```
Question: List up to 10 possible actions the state could take...
Score: -1 (requires manual evaluation)
```

## Scoring System

- **1.0**: Correct answer
- **0.0**: Incorrect answer
- **-1**: Requires manual evaluation (short answer questions)

### Scoring Logic

1. **JSON Parsing**: Extracts `answer_choice` or `answer` field from structured output
2. **Single Choice**: Exact match with correct answer
3. **Multiple Choice/Ranking**: Set comparison - all choices must match
4. **Fallback**: Plain text parsing if JSON parsing fails

## Structured Outputs

The evaluation uses Pydantic models for structured responses:

### MCQAnswer
```python
{
    "answer_choice": "B",
    "explanation": "The state violated..."
}
```

### MCQMultipleAnswer
```python
{
    "answer_choice": "A,B,C",
    "explanation": "Multiple violations..."
}
```

### ShortAnswer
```python
{
    "answer": "Detailed response..."
}
```

## Provider-Specific Notes

### OpenAI
- Uses `beta.chat.completions.parse()` for native structured outputs
- Supports all Pydantic response formats
- Fallback to JSON mode if beta API unavailable

### Anthropic
- Uses tool calling mechanism for structured outputs
- Tool spec generated from Pydantic schema
- Extracts answer from tool use blocks

### Google Gemini
- JSON schema instructions added to prompt
- Manual parsing of JSON responses
- May require prompt engineering for consistency

## Troubleshooting

### Connection Errors

**Problem:** `Error calling anthropic: Connection error`

**Solutions:**
1. Check API key is set correctly
2. Verify internet connection
3. Check firewall/proxy settings
4. Ensure API provider services are operational
5. Verify account has sufficient credits

### Invalid JSON Responses

**Problem:** Scoring fails due to unparseable responses

**Solutions:**
1. Check the `llm_response` column in CSV for actual format
2. Increase temperature for more consistent outputs
3. Update scoring logic to handle provider-specific formats

### Missing Dependencies

**Problem:** `ModuleNotFoundError`

**Solution:**
```bash
pip install -r requirements.txt
```

## File Structure

```
HumanRightsBench/
├── eval.py                 # Main evaluation script
├── model_choices.py        # LLM client wrapper
├── scenarios.json          # Benchmark dataset
├── requirements.txt        # Python dependencies
├── README.md              # This file
└── eval_results_*.csv     # Evaluation results (generated)
```

## Development

### Adding New Providers

To add support for a new LLM provider:

1. Update `ModelProvider` enum in `model_choices.py`
2. Add initialization logic in `_initialize_client()`
3. Implement provider-specific call method (e.g., `_call_newprovider()`)
4. Update `call_llm()` method to route to new provider

### Customizing Prompts

Prompts are defined in `eval.py` around lines 169-214:
- `prompt_mcq` - For multiple choice questions
- `prompt_short_answer` - For short answer questions
- `prompt_ranking` - For ranking questions

### Extending Response Formats

Add new Pydantic models at the top of `eval.py` and update the response format assignment logic around lines 220-234.

## Contributing

Contributions are welcome! Areas for improvement:
- Additional LLM provider support
- Enhanced scoring algorithms
- Better error handling and retry logic
- Batch processing optimization
- Multi-language support

## License

[Add your license information here]

## Citation

If you use this evaluation tool in your research, please cite:

```bibtex
[Add citation information here]
```

## Contact

[Add contact information here]

## Acknowledgments

This evaluation framework is part of the Human Rights Benchmark project, designed to assess LLM capabilities in understanding and reasoning about human rights issues.

