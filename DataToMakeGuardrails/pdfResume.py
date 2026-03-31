import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

# Try to import transformers for better summarization
try:
    from transformers import pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Note: Transformers not installed. Using keyword-based summarization.")
    print("For better results, install: pip install transformers torch")

def read_article_files(folder_path: str) -> Dict[str, str]:
    """Read all article txt files from the folder"""
    articles = {}
    folder = Path(folder_path)
    
    # Find all Article_*.txt files
    article_files = sorted(folder.glob("Article_*.txt"))
    
    print(f"Found {len(article_files)} article files")
    
    for file_path in article_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            articles[file_path.stem] = content
    
    return articles

def extract_keywords(text: str, top_n: int = 10) -> List[str]:
    """Extract key keywords from text for guardrail purposes"""
    # Common AI Act terms to look for
    key_terms = [
        'prohibit', 'prohibited', 'ban', 'forbidden',
        'high-risk', 'risk', 'dangerous',
        'requirement', 'obligation', 'must', 'shall',
        'transparency', 'explainable',
        'oversight', 'human', 'control',
        'data', 'privacy', 'personal',
        'biometric', 'identification',
        'compliance', 'conformity',
        'penalty', 'fine', 'sanction',
        'fundamental rights', 'safety', 'security'
    ]
    
    text_lower = text.lower()
    keywords = []
    
    for term in key_terms:
        if term in text_lower:
            # Count occurrences
            count = text_lower.count(term)
            keywords.append((term, count))
    
    # Sort by frequency and return top terms
    keywords.sort(key=lambda x: x[1], reverse=True)
    return [kw[0] for kw in keywords[:top_n]]

def extract_article_number_from_filename(filename: str) -> str:
    """Extract article number from filename like Article_1 or Article_1_1"""
    match = re.search(r'Article_(\d+(?:_\d+)?)', filename)
    if match:
        return match.group(1).replace('_', '.')
    return filename

def create_summary_with_transformers(article_text: str, max_length: int = 100) -> str:
    """Create summary using Hugging Face transformers"""
    try:
        # Initialize summarization pipeline
        summarizer = pipeline(
            "summarization", 
            model="facebook/bart-large-cnn",
            device=-1  # Use CPU
        )
        
        # Clean text
        text_to_summarize = article_text[:1024]  # Limit length for performance
        
        summary = summarizer(
            text_to_summarize,
            max_length=max_length,
            min_length=30,
            do_sample=False
        )
        
        return summary[0]['summary_text']
    except Exception as e:
        print(f"  Error in transformer summarization: {e}")
        return create_keyword_based_summary(article_text)

def create_keyword_based_summary(article_text: str, num_sentences: int = 6) -> str:
    """Create a simple keyword-based summary (fallback)"""
    
    # Clean text
    text = re.sub(r'\s+', ' ', article_text)
    text = re.sub(r'[^\w\s\.]', '', text)
    
    # Split into sentences
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    
    if not sentences:
        return "No content to summarize."
    
    # Score sentences based on important keywords
    important_keywords = [
        'shall', 'must', 'prohibit', 'require', 'obligation',
        'high-risk', 'transparency', 'oversight', 'fundamental right',
        'biometric', 'compliance', 'penalty', 'authority'
    ]
    
    sentence_scores = []
    for sentence in sentences:
        score = 0
        sentence_lower = sentence.lower()
        for keyword in important_keywords:
            if keyword in sentence_lower:
                # Keywords earlier in the article are more important
                position_score = 1.0 - (sentence_lower.find(keyword) / len(sentence_lower))
                score += (1 + position_score)
        
        # Prefer sentences that appear earlier
        position_bonus = 1.0 - (len(sentence_scores) / len(sentences))
        score += position_bonus
        
        sentence_scores.append((sentence, score))
    
    # Sort by score and take top sentences
    sentence_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Select sentences that are diverse (not too similar)
    selected_sentences = []
    for sentence, _ in sentence_scores:
        if len(selected_sentences) >= num_sentences:
            break
        
        # Avoid duplicate sentences
        if sentence not in selected_sentences:
            # Check if it's too similar to already selected sentences
            is_similar = False
            for selected in selected_sentences:
                if len(set(sentence.split()) & set(selected.split())) > len(sentence.split()) * 0.7:
                    is_similar = True
                    break
            
            if not is_similar:
                selected_sentences.append(sentence)
    
    # If we don't have enough sentences, add more from the beginning
    if len(selected_sentences) < num_sentences:
        for sentence in sentences[:num_sentences * 2]:
            if sentence not in selected_sentences and len(selected_sentences) < num_sentences:
                selected_sentences.append(sentence)
    
    return ' '.join(selected_sentences)

def extract_guardrail_rules(article_text: str, article_num: str) -> Dict:
    """Extract specific guardrail rules from article text"""
    
    rules = {
        'prohibitions': [],
        'requirements': [],
        'obligations': [],
        'rights': [],
        'penalties': []
    }
    
    text_lower = article_text.lower()
    
    # Prohibitions
    prohibition_patterns = [
        (r'(shall not|shall be prohibited|prohibited|forbidden|ban|not allowed)', 'prohibitions'),
        (r'(cannot|may not)', 'prohibitions')
    ]
    
    for pattern, category in prohibition_patterns:
        matches = re.findall(r'([^.!?]*?' + pattern + r'[^.!?]*?[.!?])', text_lower, re.IGNORECASE)
        for match in matches[:3]:  # Limit to 3 examples
            clean_match = match.strip().replace('\n', ' ')
            if clean_match not in rules[category]:
                rules[category].append(clean_match[:150])
    
    # Requirements
    requirement_patterns = [
        r'(shall\s+[^.!?]+[.!?])',
        r'(must\s+[^.!?]+[.!?])',
        r'(required\s+to\s+[^.!?]+[.!?])'
    ]
    
    for pattern in requirement_patterns:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        for match in matches[:3]:
            clean_match = match.strip().replace('\n', ' ')
            if clean_match not in rules['requirements']:
                rules['requirements'].append(clean_match[:150])
    
    # Rights
    rights_patterns = [
        r'(right to[^.!?]+[.!?])',
        r'(fundamental rights[^.!?]+[.!?])'
    ]
    
    for pattern in rights_patterns:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        for match in matches[:2]:
            clean_match = match.strip().replace('\n', ' ')
            if clean_match not in rules['rights']:
                rules['rights'].append(clean_match[:150])
    
    return rules

def create_article_resume(article_text: str, article_num: str, use_transformers: bool = False) -> str:
    """Create a concise resume (4-10 sentences) for an article"""
    
    # Extract the title/first line
    lines = article_text.strip().split('\n')
    title = lines[0] if lines else f"Article {article_num}"
    
    # Clean title
    title = re.sub(r'^\s*Article\s+\d+(?:\.\d+)?\s*', '', title, flags=re.IGNORECASE)
    title = title.strip() if title.strip() else f"Article {article_num}"
    
    # Create summary
    if use_transformers and HAS_TRANSFORMERS:
        summary = create_summary_with_transformers(article_text)
    else:
        summary = create_keyword_based_summary(article_text, num_sentences=6)
    
    # Extract guardrail rules
    rules = extract_guardrail_rules(article_text, article_num)
    
    # Format the resume
    resume = f"""=== ARTICLE {article_num} ===
Title: {title}

SUMMARY:
{summary}

KEY GUARDRAILS:
"""

    # Add key guardrail information
    if rules['prohibitions']:
        resume += "\nPROHIBITIONS:\n"
        for prohibition in rules['prohibitions'][:3]:
            resume += f"  • {prohibition}\n"
    
    if rules['requirements']:
        resume += "\nREQUIREMENTS:\n"
        for requirement in rules['requirements'][:3]:
            resume += f"  • {requirement}\n"
    
    if rules['rights']:
        resume += "\nRIGHTS PROTECTED:\n"
        for right in rules['rights'][:2]:
            resume += f"  • {right}\n"
    
    # Add key keywords
    keywords = extract_keywords(article_text, top_n=8)
    if keywords:
        resume += f"\nKEY TERMS: {', '.join(keywords)}\n"
    
    resume += "\n" + "="*60 + "\n"
    
    return resume

def create_guardrail_index(all_resumes: Dict[str, str], output_path: str):
    """Create a master index of all articles with their summaries"""
    
    index = "# EU AI Act - Article Guardrails\n\n"
    index += "This document provides summarized guardrails for each article of the EU AI Act.\n\n"
    index += "## Quick Reference by Risk Level\n\n"
    
    # Organize by categories based on article numbers
    categories = {
        'General Provisions': ['1', '2', '3', '4'],
        'Prohibited Practices': ['5'],
        'High-Risk AI Systems': ['6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27'],
        'Transparency Obligations': ['50'],
        'General-Purpose AI': ['51', '52', '53', '54', '55', '56'],
        'Innovation Measures': ['57', '58', '59', '60', '61', '62', '63'],
        'Governance': ['64', '65', '66', '67', '68', '69', '70'],
        'Post-Market Monitoring': ['72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84'],
        'Remedies': ['85', '86', '87'],
        'Penalties': ['99', '100', '101'],
        'Final Provisions': ['111', '112', '113']
    }
    
    for category, article_range in categories.items():
        index += f"### {category}\n\n"
        for article_file, resume in all_resumes.items():
            article_num = extract_article_number_from_filename(article_file)
            # Check if article number falls in this range
            for range_start in article_range:
                if article_num.startswith(range_start) or article_num.split('.')[0] in article_range:
                    # Extract first line of summary
                    summary_line = resume.split('\n')[4] if len(resume.split('\n')) > 4 else ""
                    summary_line = summary_line[:100] + "..." if len(summary_line) > 100 else summary_line
                    index += f"- **Article {article_num}**: {summary_line}\n"
                    break
        index += "\n"
    
    index += "## Detailed Article Guardrails\n\n"
    
    # Add all resumes in order
    for article_file in sorted(all_resumes.keys()):
        index += all_resumes[article_file]
    
    # Save the index
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(index)
    
    print(f"✅ Created guardrail index: {output_path}")

def main():
    # Configuration
    articles_folder = "eu_ai_act_articles"  # Folder containing article txt files
    output_folder = "guardrails"
    use_transformers = False  # Set to True if you have transformers installed
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    print("=" * 70)
    print("EU AI Act Guardrail Generator")
    print("=" * 70)
    
    # Read all article files
    print(f"\n📖 Reading articles from: {articles_folder}")
    articles = read_article_files(articles_folder)
    
    if not articles:
        print(f"\n❌ No article files found in '{articles_folder}'")
        print("   Make sure you've extracted the articles first using the extraction script.")
        return
    
    print(f"\n📝 Creating resumes for {len(articles)} articles...")
    
    all_resumes = {}
    
    for article_file, content in articles.items():
        article_num = extract_article_number_from_filename(article_file)
        print(f"  Processing {article_file}...", end=" ")
        
        # Create resume
        resume = create_article_resume(content, article_num, use_transformers)
        all_resumes[article_file] = resume
        
        # Save individual resume
        output_file = os.path.join(output_folder, f"{article_file}_guardrail.txt")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(resume)
        
        print("✓")
    
    # Create master index
    print(f"\n📚 Creating master guardrail index...")
    create_guardrail_index(all_resumes, os.path.join(output_folder, "00_MASTER_GUARDRAIL_INDEX.txt"))
    
    # Create a simplified JSON version for programmatic use
    import json
    simplified = {}
    for article_file, resume in all_resumes.items():
        article_num = extract_article_number_from_filename(article_file)
        # Extract summary from resume
        summary_match = re.search(r'SUMMARY:\n(.*?)\n\nKEY GUARDRAILS:', resume, re.DOTALL)
        summary = summary_match.group(1).strip() if summary_match else ""
        
        # Extract key terms
        terms_match = re.search(r'KEY TERMS: (.*?)\n', resume)
        terms = terms_match.group(1).split(', ') if terms_match else []
        
        simplified[article_num] = {
            'summary': summary,
            'key_terms': terms
        }
    
    json_path = os.path.join(output_folder, "guardrails.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(simplified, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Created JSON guardrail file: {json_path}")
    
    print("\n" + "=" * 70)
    print("✅ Guardrail generation complete!")
    print(f"   Individual guardrails: {output_folder}/")
    print(f"   Master index: {output_folder}/00_MASTER_GUARDRAIL_INDEX.txt")
    print(f"   JSON format: {output_folder}/guardrails.json")
    print("=" * 70)

if __name__ == "__main__":
    main()