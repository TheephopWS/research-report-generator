import json
import os
from pathlib import Path
from glob import glob
from deepeval.models import DeepSeekModel
from deepeval.metrics import ContextualRelevancyMetric, GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from dotenv import load_dotenv
load_dotenv()

deepseek_model = DeepSeekModel(
    model="deepseek-chat", # This is a more powerful model optimized for reasoning tasks
    temperature=0,
    api_key=os.environ.get("DEEPSEEK_API_KEY")
)

relevancy_metric = ContextualRelevancyMetric(threshold=0.7, model=deepseek_model, verbose_mode=True)

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
    
    # Create test case
    test_case = LLMTestCase(
        input=input_text,
        actual_output=actual_output,
        retrieval_context=retrieval_context if isinstance(retrieval_context, list) else [retrieval_context]
    )
    
    # Measure metrics
    relevancy_metric.measure(test_case)
    
    result = {
        "file": os.path.basename(context_file),
        "relevancy_score": relevancy_metric.score
    }
    results.append(result)
    
    print(f"  Relevancy Score: {relevancy_metric.score}")

# Print summary
print("\n=== SUMMARY ===")
print(f"On average, the relevancy score across all reports is: {sum(r['relevancy_score'] for r in results) / len(results):.2f}")