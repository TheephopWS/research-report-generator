import re
import time
import requests
import json
import os
from pathlib import Path
from dataclasses import dataclass


@dataclass
class CitationResult:
    raw_reference: str
    parsed_title: str
    found_in_semantic_scholar: bool
    found_in_crossref: bool
    is_valid: bool
    matched_title: str = ""
    source: str = ""


def parse_title_from_reference(reference: str) -> str:
    quoted = re.search(r'["""](.+?)["""]', reference)
    if quoted:
        return quoted.group(1).strip()

    match = re.search(r'\(\d{4}\)\.\s*(.+?)\.', reference)
    if match:
        return match.group(1).strip()

    match = re.search(r',\s*\d{4}\.\s*(.+?)\.', reference)
    if match:
        return match.group(1).strip()

    parts = re.split(r'\.\s+', reference)
    if len(parts) >= 2 and len(parts[1].strip()) > 10:
        return parts[1].strip()

    return reference[:150]


def title_similarity(title_a: str, title_b: str) -> float:
    clean = lambda s: set(re.sub(r'[^\w\s]', '', s).lower().split())
    tokens_a = clean(title_a)
    tokens_b = clean(title_b)
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def check_semantic_scholar(title: str, max_retries: int = 3) -> tuple[bool, str]:
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {"query": title, "limit": 3, "fields": "title"}

    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 429:
                time.sleep(2 ** (attempt + 1))
                continue
            if response.status_code == 200:
                data = response.json()
                for paper in data.get("data", []):
                    if paper.get("title"):
                        if title_similarity(title.lower(), paper["title"].lower()) > 0.75:
                            return True, paper["title"]
                return False, ""
        except requests.RequestException:
            time.sleep(1)
    return False, ""


def check_crossref(title: str, max_retries: int = 3) -> tuple[bool, str]:
    url = "https://api.crossref.org/works"
    params = {"query.title": title, "rows": 3, "select": "title"}
    headers = {"User-Agent": "CitationValidator/1.0 (mailto:your_email@example.com)"}

    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=10)
            if response.status_code == 200:
                items = response.json().get("message", {}).get("items", [])
                for item in items:
                    if item.get("title"):
                        candidate = item["title"][0]
                        if title_similarity(title.lower(), candidate.lower()) > 0.75:
                            return True, candidate
                return False, ""
        except requests.RequestException:
            time.sleep(1)
    return False, ""


def validate_citations(references: list[str], delay: float = 1.0) -> dict:
    """
    Takes a list of reference strings and validates each one.

    Args:
        references: List of reference strings, e.g.
                    ["Vaswani et al. (2017). Attention is All You Need. NeurIPS.",
                     "Devlin et al. (2019). BERT: Pre-training of ... NAACL."]
        delay: Seconds between API calls.

    Returns:
        Dictionary with CVR score and per-citation details.
    """
    print(f"Validating {len(references)} references.\n")
    results = []

    for i, ref in enumerate(references):
        title = parse_title_from_reference(ref)
        print(f"[{i+1}/{len(references)}] Checking: {title}")

        ss_found, ss_title = check_semantic_scholar(title)
        time.sleep(delay)

        cr_found, cr_title = check_crossref(title)
        time.sleep(delay)

        is_valid = ss_found or cr_found
        matched = ss_title if ss_found else cr_title
        if ss_found and cr_found:
            source = "both"
        elif ss_found:
            source = "semantic_scholar"
        elif cr_found:
            source = "crossref"
        else:
            source = "not_found"

        results.append(CitationResult(
            raw_reference=ref,
            parsed_title=title,
            found_in_semantic_scholar=ss_found,
            found_in_crossref=cr_found,
            is_valid=is_valid,
            matched_title=matched,
            source=source,
        ))
        print(f"  -> {'VALID' if is_valid else 'NOT FOUND'} ({source})\n")

    valid_count = sum(1 for r in results if r.is_valid)
    total_count = len(results)
    cvr = valid_count / total_count if total_count > 0 else 0.0

    print("=" * 60)
    print(f"CVR: {valid_count}/{total_count} = {cvr:.4f}")
    print("=" * 60)

    return {"cvr": cvr, "valid_count": valid_count, "total_count": total_count, "details": results}


def extract_references_from_output(output_text: str) -> list[str]:
    """
    Extracts references from the References section of the output text.
    Assumes references are numbered list items starting with "## References".
    
    Returns:
        List of individual reference strings.
    """
    # Find the References section
    references_match = re.search(r'##\s*References\s*\n(.*)', output_text, re.DOTALL)
    if not references_match:
        return []
    
    references_section = references_match.group(1).strip()
    
    # Split by numbered items (e.g., "1. ", "2. ", etc.)
    # This regex matches lines starting with a number followed by a period and space
    reference_items = re.split(r'\n(?=\d+\.)\s*', references_section)
    
    # Clean up each reference (remove leading number and period)
    references = []
    for item in reference_items:
        item = re.sub(r'^\d+\.\s*', '', item.strip())
        if item and len(item) > 10:  # Filter out short/empty items
            references.append(item)
    
    return references


def load_contexts_from_json_files(contexts_dir: str) -> list[dict]:
    """
    Loads all JSON files from the specified directory.
    
    Returns:
        List of dictionaries with keys: 'file', 'input', 'output', 'references'
    """
    all_contexts = []
    
    if not os.path.isdir(contexts_dir):
        print(f"Error: Directory {contexts_dir} does not exist.")
        return all_contexts
    
    json_files = sorted(Path(contexts_dir).glob('*.json'))
    print(f"Found {len(json_files)} JSON files in {contexts_dir}\n")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                output = data.get('output', '')
                references = extract_references_from_output(output)
                
                all_contexts.append({
                    'file': json_file.name,
                    'input': data.get('input', ''),
                    'output': output,
                    'references': references
                })
                
        except Exception as e:
            print(f"Error reading {json_file.name}: {e}")
    
    return all_contexts


def evaluate_all_contexts(contexts_dir: str, delay: float = 1.0) -> dict:
    """
    Evaluates all contexts in the directory and calculates CVR for each.
    
    Returns:
        Dictionary with aggregated CVR, per-file results, and statistics.
    """
    contexts = load_contexts_from_json_files(contexts_dir)
    
    if not contexts:
        print("No contexts loaded.")
        return {}
    
    all_results = {
        'contexts_evaluated': len(contexts),
        'per_file_results': [],
        'aggregate_stats': {
            'total_references': 0,
            'total_valid': 0,
            'overall_cvr': 0.0,
            'average_cvr': 0.0,
            'min_cvr': 1.0,
            'max_cvr': 0.0,
        }
    }
    
    total_valid = 0
    total_references = 0
    cvr_scores = []
    
    for context in contexts:
        print(f"\n{'='*60}")
        print(f"File: {context['file']}")
        print(f"Input: {context['input'][:80]}...")
        print(f"References found: {len(context['references'])}")
        print(f"{'='*60}\n")
        
        if context['references']:
            file_results = validate_citations(context['references'], delay=delay)
            
            file_cvr = file_results['cvr']
            cvr_scores.append(file_cvr)
            total_valid += file_results['valid_count']
            total_references += file_results['total_count']
            
            all_results['per_file_results'].append({
                'file': context['file'],
                'input': context['input'],
                'cvr': file_cvr,
                'valid_count': file_results['valid_count'],
                'total_count': file_results['total_count'],
                'details': [
                    {
                        'reference': r.raw_reference[:100],
                        'is_valid': r.is_valid,
                        'source': r.source
                    }
                    for r in file_results['details']
                ]
            })
        else:
            print(f"Warning: No references found in {context['file']}\n")
    
    # Calculate aggregate statistics
    if cvr_scores:
        all_results['aggregate_stats']['total_references'] = total_references
        all_results['aggregate_stats']['total_valid'] = total_valid
        all_results['aggregate_stats']['overall_cvr'] = total_valid / total_references if total_references > 0 else 0.0
        all_results['aggregate_stats']['average_cvr'] = sum(cvr_scores) / len(cvr_scores)
        all_results['aggregate_stats']['min_cvr'] = min(cvr_scores)
        all_results['aggregate_stats']['max_cvr'] = max(cvr_scores)
    
    print(f"\n\n{'='*60}")
    print("AGGREGATE STATISTICS")
    print(f"{'='*60}")
    print(f"Total References: {all_results['aggregate_stats']['total_references']}")
    print(f"Total Valid: {all_results['aggregate_stats']['total_valid']}")
    print(f"Overall CVR: {all_results['aggregate_stats']['overall_cvr']:.4f}")
    print(f"Average CVR: {all_results['aggregate_stats']['average_cvr']:.4f}")
    print(f"Min CVR: {all_results['aggregate_stats']['min_cvr']:.4f}")
    print(f"Max CVR: {all_results['aggregate_stats']['max_cvr']:.4f}")
    print(f"Files Evaluated: {all_results['contexts_evaluated']}")
    print(f"{'='*60}\n")
    
    return all_results


if __name__ == "__main__":
    # Get the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    contexts_dir = os.path.join(script_dir, '..', 'contexts_outputs')
    
    # Evaluate all contexts
    results = evaluate_all_contexts(contexts_dir, delay=1.0)

"""
if __name__ == "__main__":
    
    references = [
        'Vaswani, A., Shazeer, N., et al. (2017). Attention is All You Need. NeurIPS.',
        'Devlin, J., Chang, M., Lee, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers. NAACL.',
        'Smith, J. (2023). A Completely Fabricated Paper That Does Not Exist. Fake Conference.',
    ]
    
    references = [
        'Wilson, Peter H. *The Thirty Years War: Europe’s Tragedy*. Belknap Press, 2009.',
        'Wilson, Peter H. *The Thirty Years War: A Brief History with Documents*. Hackett Publishing Company, 2018.',
        'Helfferich, Tryntje. "The Causes of the Thirty Years War: A Review of Recent Historiography." *The Historian*, vol. 69, no. 4, 2007, pp. 703–728. doi:10.1111/j.1540-6563.2007.00189.x',
        'Parker, Geoffrey. *Global Crisis: War, Climate Change and Catastrophe in the Seventeenth Century*. Yale University Press, 2013.',
        'Parrott, David. *The Business of War: Military Enterprise and Military Revolution in Early Modern Europe*. Cambridge University Press, 2012.',
    ]


    results = validate_citations(references)
"""