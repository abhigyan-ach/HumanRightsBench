import pandas as pd
import json
import argparse
from pydantic import BaseModel
from typing import Literal
from model_choices import get_llm_client, LLMClient

class MCQAnswer(BaseModel):
    """Structured output for single-choice MCQ questions"""
    answer_choice: Literal['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    explanation: str

class MCQMultipleAnswer(BaseModel):
    """Structured output for multiple-choice MCQ questions (e.g., 'A,B,C')"""
    answer_choice: str  # Could be "A,B,C" or "A"
    explanation: str

class ShortAnswer(BaseModel):
    """Structured output for short answer questions"""
    answer: str


def parse_args():
    """Parse command line arguments for model selection"""
    parser = argparse.ArgumentParser(description='Evaluate LLMs on human rights scenarios')
    parser.add_argument('--provider', type=str, required=True, 
                       choices=['openai', 'anthropic', 'google', 'qwen'],
                       help='LLM provider to use')
    parser.add_argument('--model', type=str, required=True,
                       help='Model name (e.g., gpt-4, claude-3-5-sonnet-20241022, gemini-pro)')
    parser.add_argument('--temperature', type=float, default=0.0,
                       help='Sampling temperature (default: 0.0)')
    parser.add_argument('--max-tokens', type=int, default=1000,
                       help='Maximum tokens in response (default: 1000)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Print prompts without calling API')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of evaluations to run (for testing)')
    parser.add_argument('--api_key', type=str, default=None )
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # Initialize LLM client (unless dry run)
    llm_client = get_llm_client(
        provider=args.provider,
        model_name=args.model,
        temperature=args.temperature,
        # max_tokens=args.max_tokens,
        api_key= args.api_key
    )
    
    print(f"Running evaluation with:")
    print(f"  Provider: {args.provider}")
    print(f"  Model: {args.model}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Dry run: {args.dry_run}")
    print()
    
    with open('scenarios.json', 'r') as f:
        scenarios_data = json.load(f)

    data = []
    
    # Create a mapping of question_id to question_text for easy lookup
    question_map = {q['question_id']: q['question_text'] 
                    for q in scenarios_data['dataset_metadata']['evaluation_questions']}
    
    for scenario in scenarios_data['scenarios']:
        scenario_id = scenario['scenario_id']
        scenario_text = scenario['scenario_text']
        tags = scenario['scenario_metadata']['tags']
        difficulty = scenario['scenario_metadata']['difficulty']
        
        for subscenario in scenario['subscenarios']:
            subscenario_id = subscenario['subscenario_id']
            subscenario_text = subscenario['subscenario_text']
            
            # Each subscenario has multiple evaluations (questions)
            for evaluation in subscenario['evaluations']:
                question_id = evaluation['question_id']
                question_text = question_map.get(question_id, '')
                answer_choices = evaluation.get('answer_choices', [])
                correct_answer = evaluation.get('correct_answer', '')
                
                # Create a row for each evaluation
                row = [
                    scenario_id,
                    scenario_text,
                    tags,
                    difficulty,
                    subscenario_id,
                    subscenario_text,
                    question_id,
                    question_text,
                    answer_choices,
                    correct_answer,
                    '',  # llm_response placeholder
                    ''   # score placeholder
                ]
                data.append(row)
    
    # Convert to DataFrame
    df = pd.DataFrame(data, columns=[
        'scenario_id',
        'scenario_text',
        'tags',
        'difficulty',
        'subscenario_id',
        'subscenario_text',
        'question_id',
        'question_text',
        'answer_choices',
        'correct_answer',
        'llm_response',
        'score'
    ])

    # Function to format answer choices nicely
    def format_answer_choices(choices):
        """Format answer choices list into a readable string"""
        if not choices or len(choices) == 0:
            return "This is a short answer question with no multiple choice options."
        
        formatted = []
        for choice in choices:
            choice_id = choice.get('choice_id', '')
            choice_text = choice.get('choice_text', '')
            formatted.append(f"{choice_id}. {choice_text}")
        
        return "\n".join(formatted)
    
    def calculate_score(llm_response: str, correct_answer: str, question_id: str) -> float:
        """Calculate score for the LLM response"""
        # For short answer questions (P), we'll need manual evaluation
        if question_id == 'P':
            #add functinality for 
            return -1  # Mark for manual evaluation
        
        # Parse JSON response if it's structured output
        llm_answer = llm_response.strip()
        
        # Try to parse as JSON first (for structured outputs)
        try:
            parsed_response = json.loads(llm_answer)
            if 'answer_choice' in parsed_response:
                llm_answer = parsed_response['answer_choice'].strip().upper()
            elif 'answer' in parsed_response:
                llm_answer = parsed_response['answer'].strip().upper()
            else:
                llm_answer = llm_answer.upper()
        except (json.JSONDecodeError, TypeError):
            # If not JSON, treat as plain text
            llm_answer = llm_answer.upper()
        
        # Try to extract answer letter (A, B, C, etc.)
        if ',' in correct_answer:  # Multiple answers like "A,B,C"
            # For ranking questions - exact match required
            correct_set = set(correct_answer.split(','))
            llm_set = set([c for c in llm_answer if c in 'ABCDEFGH'])
            return 1.0 if llm_set == correct_set else 0.0
        else:
            # Single answer: check if the single letter answer is the correct answer:
            return 1.0 if correct_answer.upper() == llm_answer else 0.0
    
    # Iterate through the dataframe
    total_rows = len(df) if args.limit is None else min(args.limit, len(df))
    
    for idx, row in df.iterrows():
        if args.limit and idx >= args.limit:
            break
            
        # Format the answer choices
        formatted_choices = format_answer_choices(row['answer_choices'])
        
        # Build the prompt
        prompt_mcq = f"""You are an expert in human rights. You are given a scenario of possible human rights violations and a question.
            SCENARIO:
            {row['scenario_text']}

            SUBSCENARIO:
            {row['subscenario_text']}

            QUESTION:
            {row['question_text']}

            ANSWER CHOICES:
            {formatted_choices}

            Please provide your answer with the letter choice and a brief explanation."""



        prompt_short_answer = f"""You are an expert in human rights. You are given a scenario of possible human rights violations and a question that requires a short answer.
            SCENARIO:
            {row['scenario_text']}

            SUBSCENARIO:
            {row['subscenario_text']}

            QUESTION:
            {row['question_text']}

            ANSWER CHOICES:
            {formatted_choices}

            Please provide your answer as a single letter, as per the JSON format."""

        prompt_ranking = f"""You are an expert in human rights. You are given a scenario of possible human rights violations and a question.
            SCENARIO:
            {row['scenario_text']}

            SUBSCENARIO:
            {row['subscenario_text']}

            QUESTION:
            {row['question_text']}

            ANSWER CHOICES:
            {formatted_choices}

            Please provide your answer as a ranking of the answer choices: do this as a sequence of letters: For example if the correct ranking is A:rank1 B:rank2 C:rank3 then return "A,B,C". The answer should be a single string, as per the JSON format."""
        
        print(f"\n{'='*80}")
        print(f"Progress: {idx + 1}/{total_rows} - Question ID: {row['question_id']}")
        print(f"{'='*80}")
        
        prompt=""
        response_format=None

        if row['question_id'] == 'P':
            print(prompt_short_answer)
            prompt=prompt_short_answer
            response_format=ShortAnswer
        elif row['question_id'] == 'R':
            print(prompt_ranking)
            prompt=prompt_ranking
            response_format=MCQMultipleAnswer
        else:
            print(prompt_mcq)
            prompt=prompt_mcq
            response_format=MCQAnswer

        
            # Call LLM API
        try:
                llm_response = llm_client.call_llm(prompt, response_format=response_format)
                print(f"LLM Response: {llm_response}")
                print(f"Correct Answer: {row['correct_answer']}")
                
                # Calculate score
                score = calculate_score(llm_response, row['correct_answer'], row['question_id'])
                print(f"Score: {score}")
                
                # Store results
                df.at[idx, 'llm_response'] = llm_response
                df.at[idx, 'response_format'] = response_format
                df.at[idx, 'prompt'] = prompt
                df.at[idx, 'score'] = score
                
        except Exception as e:
                print(f"Error calling LLM: {e}")
                df.at[idx, 'llm_response'] = f"ERROR: {str(e)}"
                df.at[idx, 'score'] = -1
    
    # Save the dataframe with results
    if not args.dry_run:
        output_file = f'eval_results_{args.provider}_{args.model.replace("/", "_")}.csv'
        df.to_csv(output_file, index=False)
        print(f"\n{'='*80}")
        print(f"Results saved to: {output_file}")
        
        # Print summary statistics
        scored_rows = df[pd.to_numeric(df['score'], errors='coerce') >= 0]
        if len(scored_rows) > 0:
            avg_score = scored_rows['score'].mean()
            print(f"Average Score: {avg_score:.2%}")
            print(f"Total Questions: {len(scored_rows)}")
            print(f"Correct: {(scored_rows['score'] == 1.0).sum()}")
            print(f"Incorrect: {(scored_rows['score'] == 0.0).sum()}")
        print(f"{'='*80}")
    
