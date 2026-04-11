import json
import os
from pathlib import Path
from glob import glob
from deepeval.models import DeepSeekModel
from deepeval.metrics import AnswerRelevancyMetric, GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from dotenv import load_dotenv
load_dotenv()

# 1. Define DeepSeek as your custom model
# base_url for DeepSeek is usually: https://api.deepseek.com
deepseek_model = DeepSeekModel(
    model="deepseek-chat", # This is a more powerful model optimized for reasoning tasks
    temperature=0,
    api_key=os.environ.get("DEEPSEEK_API_KEY")
)

# 2. Use this model in your metrics
coherence_metric = GEval(
    name="Coherence",
    evaluation_steps=[
        "Check for logical contradictions: Does the report contain any statements that contradict each other? List any contradictions found.",
        "Check for structural coherence: Are ideas organized in a logical sequence? Does the conclusion follow from earlier arguments? Identify any logical gaps.",
        "Check for consistency of terminology and concepts: Are key terms used consistently throughout? Are concepts clearly defined before use?",
        "Check for supporting evidence: Do claims have sufficient supporting information? Are assertions made without evidence?",
        "Based on the above analysis, assign a score from 0 to 1 where: 1.0 = no contradictions and perfect flow, 0.9 =almost perfect, 0.7-0.89 = minor issues but mostly coherent, 0.5-0.69 = some logical gaps or inconsistencies, 0.0-0.49 = significant coherence problems. Return ONLY the single number (e.g., 0.75)."
    ],
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    model=deepseek_model
)

fluency_metric = GEval(
    name="Fluency",
    evaluation_steps=[
        "Check if the sentences flow smoothly and are easy to read",
        "Evaluate whether the report is grammatically correct and well-structured",
        "Identify any awkward phrasing or unnatural language that hinders readability."
    ],
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    model=deepseek_model
)

relevancy_metric = AnswerRelevancyMetric(threshold=0.7, model=deepseek_model)

# 3. Load and evaluate context_{i}.json files
contexts_dir = Path(__file__).parent.parent / "contexts_outputs"
context_files = sorted(glob(str(contexts_dir / "contexts_*.json")), 
                       key=lambda x: int(x.split("_")[-1].split(".")[0]))

results = []

for context_file in context_files:
    print(f"\nEvaluating {os.path.basename(context_file)}...")
    
    with open(context_file, 'r') as f:
        context_data = json.load(f)
    
    # Adjust field names based on your actual JSON structure
    input_text = context_data.get("input", "")
    actual_output = context_data.get("output", "")
    retrieval_context = context_data.get("contexts", [])

    # Validate and convert retrieval_context to list of strings
    if not isinstance(retrieval_context, list):
        retrieval_context = [retrieval_context]
    
    # Convert all items to strings to prevent API overload
    MAX_CONTEXT_LENGTH = 1000  # Limit each context chunk
    CHUNK_SIZE = 2000          # Size of each output chunk to evaluate
    
    retrieval_context = [str(item)[:MAX_CONTEXT_LENGTH] if not isinstance(item, str) else item[:MAX_CONTEXT_LENGTH] for item in retrieval_context]
    retrieval_context = [item for item in retrieval_context if item and item.strip()]  # Remove empty strings
    
    # Skip if missing critical data
    if not input_text or not actual_output or not retrieval_context:
        continue
    
    # Split actual_output into chunks and evaluate each
    from time import sleep
    output_chunks = [actual_output[i:i+CHUNK_SIZE] for i in range(0, len(actual_output), CHUNK_SIZE)]
    chunk_scores = []
    max_retries = 3
    
    print(f"Evaluating {os.path.basename(context_file)} ({len(output_chunks)} chunks)...")
    
    for chunk_idx, output_chunk in enumerate(output_chunks, 1):
        retry_count = 0
        chunk_score = None
        
        while retry_count < max_retries:
            try:
                # Create test case for this chunk
                test_case = LLMTestCase(
                    input=input_text,
                    actual_output=output_chunk,
                    retrieval_context=retrieval_context
                )
                
                # Measure relevancy for this chunk
                relevancy_metric.measure(test_case)
                chunk_score = relevancy_metric.score
                chunk_scores.append(chunk_score)
                print(f"  Chunk {chunk_idx}/{len(output_chunks)}: {chunk_score:.2f}")
                break
            except Exception as e:
                retry_count += 1
                if retry_count < max_retries:
                    wait_time = 2 ** retry_count
                    sleep(wait_time)
                else:
                    print(f"  Chunk {chunk_idx}/{len(output_chunks)}: Failed after {max_retries} retries")
                    break
        
        if chunk_score is None:
            # If a chunk fails, skip this file
            break
    
    if not chunk_scores or len(chunk_scores) < len(output_chunks):
        continue
    
    # Average the chunk scores
    relevancy_score = sum(chunk_scores) / len(chunk_scores)
    
    result = {
        "file": os.path.basename(context_file),
        "relevancy_score": relevancy_score,
    }
    results.append(result)

# Print summary
print("\n=== SUMMARY ===")
print(f"On average, the relevancy score across all reports is: {sum(r['relevancy_score'] for r in results) / len(results):.2f}")
"""
for result in results:
    print(f"{result['file']}: Coherence={result['coherence_score']:.2f}, Relevancy={result['relevancy_score']:.2f}")
"""