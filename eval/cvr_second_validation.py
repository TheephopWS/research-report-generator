import json
import re
import requests
import time
from difflib import SequenceMatcher

# Configuration
EMAIL = "ptrakarnk2-c@my.cityu.edu.hk"  # Enter your email for the OpenAlex Polite Pool
SIMILARITY_THRESHOLD = 0.75

def title_similarity(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def extract_doi(text):
    """Extracts DOI using regex pattern."""
    doi_pattern = r"10\.\d{4,9}/[-._;()/:A-Z0-9]+"
    match = re.search(doi_pattern, text, re.IGNORECASE)
    return match.group(0) if match else None

def is_website(text):
    """Checks if reference points to a major report site or contains a URL."""
    web_indicators = [".org", ".gov", ".int", "http", "Available via"]
    return any(indicator in text for indicator in web_indicators)

def validate_citations(json_data):
    results = {
        "perfect_match": 0,
        "reconciled": 0,
        "failed": 0,
        "details": []
    }
    
    citations = json_data.get("invalid_citations", [])
    
    for cite in citations:
        raw = cite["raw_reference"]
        title = cite["parsed_title"]
        doi = extract_doi(raw)
        
        # 1. Check for Website (Count as Perfect Match per instructions)
        if is_website(raw) and not doi:
            results["perfect_match"] += 1
            results["details"].append({"title": title, "status": "Perfect Match (Website)"})
            continue

        # 2. Check DOI (Reconciliation Logic)
        if doi:
            url = f"https://api.openalex.org/works/https://doi.org/{doi}?mailto={EMAIL}"
            try:
                resp = requests.get(url, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    canonical_title = data.get("title", "")
                    
                    # If DOI is valid, it's Reconciled (even if title was inaccurate)
                    results["reconciled"] += 1
                    results["details"].append({
                        "title": title, 
                        "canonical": canonical_title, 
                        "status": "Reconciled (DOI)"
                    })
                    continue
            except Exception:
                pass

        # 3. Fuzzy Validation (Count as Perfect Match per instructions)
        # Search OpenAlex by title
        search_url = f"https://api.openalex.org/works?filter=title.search:{title}&mailto={EMAIL}"
        try:
            search_resp = requests.get(search_url, timeout=10)
            if search_resp.status_code == 200:
                top_result = search_resp.json().get("results", [])
                if top_result:
                    candidate_title = top_result[0].get("title", "")
                    if title_similarity(title, candidate_title) > SIMILARITY_THRESHOLD:
                        results["perfect_match"] += 1
                        results["details"].append({"title": title, "status": "Perfect Match (Fuzzy)"})
                        continue
        except Exception:
            pass

        # 4. If all else fails
        results["failed"] += 1
        results["details"].append({"title": title, "status": "Failed"})
        time.sleep(0.1) # Polite delay

    return results

# Example Usage
with open('invalid_citations.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

stats = validate_citations(data)

# Calculate Rates
total = len(data["invalid_citations"])
perfect_rate = (stats["perfect_match"] / total) * 100
reconcile_rate = (stats["reconciled"] / total) * 100

print(f"--- Evaluation Results ---")
print(f"Perfect Match Rate: {perfect_rate:.2f}%")
print(f"Reconciliation Rate: {reconcile_rate:.2f}%")
print(f"Total Validated: {perfect_rate + reconcile_rate:.2f}%")