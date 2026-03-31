import pdfplumber
import re
import os

def extract_articles_robust(pdf_path, output_folder="eu_ai_act_articles"):
    """Robust article extraction that handles the EU AI Act structure"""
    
    os.makedirs(output_folder, exist_ok=True)
    
    print("Reading PDF and extracting text...")
    full_text = ""
    
    # Extract all text first
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            text = page.extract_text()
            if text:
                full_text += text + "\n"
            if page_num % 20 == 0:
                print(f"  Processed page {page_num}/{len(pdf.pages)}")
    
    print(f"Total extracted text length: {len(full_text)} characters")
    
    # Look for articles using multiple patterns
    print("\nSearching for articles...")
    
    # Method 1: Find all "Article X" occurrences with their positions
    article_positions = []
    for match in re.finditer(r'\nArticle\s+(\d+(?:\.\d+)?)\s+', full_text):
        article_num = match.group(1)
        start_pos = match.start()
        article_positions.append((article_num, start_pos))
    
    print(f"Found {len(article_positions)} article markers")
    
    if len(article_positions) == 0:
        # Try alternative pattern - articles might be at line starts
        for match in re.finditer(r'^Article\s+(\d+(?:\.\d+)?)\s+', full_text, re.MULTILINE):
            article_num = match.group(1)
            start_pos = match.start()
            article_positions.append((article_num, start_pos))
    
    print(f"Found {len(article_positions)} total article markers")
    
    if len(article_positions) == 0:
        print("\nTrying to find articles with a different approach...")
        # Look for ANY Article pattern
        simple_matches = re.findall(r'Article\s+(\d+)', full_text)
        print(f"Found {len(simple_matches)} Article references")
        
        # Let's also print a sample of the text to debug
        print("\nSample of text (first 1000 chars):")
        print(full_text[:1000])
        
        return []
    
    # Extract articles based on positions
    articles = []
    for i, (article_num, start_pos) in enumerate(article_positions):
        # Determine end position (start of next article or end of text)
        if i + 1 < len(article_positions):
            end_pos = article_positions[i + 1][1]
        else:
            end_pos = len(full_text)
        
        # Extract article text
        article_text = full_text[start_pos:end_pos].strip()
        
        # Clean up the article text
        # Remove page markers
        article_text = re.sub(r'={5,} Page \d+ ={5,}', '', article_text)
        article_text = re.sub(r'ELI: http://[^\n]+', '', article_text)
        article_text = re.sub(r'\n\d+/\d+\n', '\n', article_text)
        article_text = re.sub(r'^EN\n', '', article_text, flags=re.MULTILINE)
        article_text = re.sub(r'OJ L,.*?\n', '', article_text)
        
        # Remove extra whitespace
        article_text = re.sub(r'\n{3,}', '\n\n', article_text)
        
        articles.append((article_num, article_text))
    
    print(f"\nExtracting {len(articles)} articles...")
    
    # Save articles
    for article_num, article_text in articles:
        # Create filename (replace dots with underscores for sub-articles)
        safe_num = article_num.replace('.', '_')
        filename = f"Article_{safe_num}.txt"
        filepath = os.path.join(output_folder, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(article_text)
        
        print(f"  Created: {filename}")
    
    print(f"\n✅ Successfully extracted {len(articles)} articles to '{output_folder}/'")
    return articles

def extract_articles_by_line(pdf_path, output_folder="eu_ai_act_articles"):
    """Alternative approach: process page by page and detect articles by line"""
    
    os.makedirs(output_folder, exist_ok=True)
    
    print("Processing page by page...")
    
    current_article = None
    article_content = {}
    in_article = False
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            text = page.extract_text()
            if not text:
                continue
            
            lines = text.split('\n')
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check if this line starts an article
                # Look for "Article X" at the beginning of line
                article_match = re.match(r'^Article\s+(\d+(?:\.\d+)?)(?:\s+|$)', line)
                
                if article_match:
                    # Save previous article if exists
                    if current_article and current_article in article_content:
                        pass  # Continue accumulating
                    
                    article_num = article_match.group(1)
                    current_article = article_num
                    
                    if current_article not in article_content:
                        article_content[current_article] = []
                    
                    # Add this line as the start
                    article_content[current_article].append(line)
                    in_article = True
                    
                elif in_article and current_article:
                    # Skip obvious footer/header lines
                    if not re.match(r'={5,} Page \d+ ={5,}', line) and \
                       not re.match(r'ELI:', line) and \
                       not re.match(r'^\d+/\d+$', line) and \
                       not re.match(r'^OJ L,', line) and \
                       not re.match(r'^EN$', line):
                        article_content[current_article].append(line)
            
            if page_num % 50 == 0:
                print(f"  Processed page {page_num}/{len(pdf.pages)}")
    
    print(f"\nFound {len(article_content)} articles")
    
    if len(article_content) == 0:
        print("\nNo articles found. Debug info:")
        print("First 5 pages of text to understand the structure:")
        
        with pdfplumber.open(pdf_path) as pdf:
            for i in range(min(5, len(pdf.pages))):
                text = pdf.pages[i].extract_text()
                print(f"\n--- Page {i+1} first 500 chars ---")
                print(text[:500] if text else "No text")
        
        return []
    
    # Save articles
    for article_num, lines in article_content.items():
        article_text = '\n'.join(lines)
        
        # Clean up
        article_text = re.sub(r'={5,} Page \d+ ={5,}', '', article_text)
        
        safe_num = article_num.replace('.', '_')
        filename = f"Article_{safe_num}.txt"
        filepath = os.path.join(output_folder, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(article_text)
        
        print(f"  Created: {filename}")
    
    print(f"\n✅ Successfully extracted {len(article_content)} articles to '{output_folder}/'")
    return article_content

def manual_parse_articles(pdf_path, output_folder="eu_ai_act_articles"):
    """
    Manually parse articles by looking for the pattern in the text
    based on the actual structure seen in the PDF
    """
    
    os.makedirs(output_folder, exist_ok=True)
    
    print("Reading PDF...")
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    
    print(f"Text length: {len(text)} characters")
    
    # Look for the chapter structure first
    chapters = re.findall(r'# CHAPTER ([IVX]+|[A-Z]+)\s+(.*?)(?=\n# CHAPTER|\nArticle\s+\d+|\Z)', text, re.DOTALL)
    print(f"Found {len(chapters)} chapters")
    
    # Look for articles with their titles
    # Pattern based on actual PDF: "Article X\nTitle" or "Article X Title"
    article_pattern = r'\nArticle\s+(\d+(?:\.\d+)?)\s*\n\s*([A-Z][^\n]*?)(?:\n|$)'
    
    articles = re.findall(article_pattern, text)
    print(f"Found {len(articles)} articles with pattern 1")
    
    if len(articles) == 0:
        # Try different pattern: "Article X Title" on same line
        article_pattern2 = r'\nArticle\s+(\d+(?:\.\d+)?)\s+([A-Z][^\n]*)'
        articles = re.findall(article_pattern2, text)
        print(f"Found {len(articles)} articles with pattern 2")
    
    if len(articles) == 0:
        # Try to find all "Article" occurrences and extract the following text
        all_articles = []
        for match in re.finditer(r'Article\s+(\d+(?:\.\d+)?)', text):
            article_num = match.group(1)
            start = match.start()
            
            # Find where next article starts
            next_article = re.search(r'Article\s+\d+(?:\.\d+)?', text[start + len(match.group()):])
            if next_article:
                end = start + len(match.group()) + next_article.start()
            else:
                end = len(text)
            
            article_text = text[start:end].strip()
            all_articles.append((article_num, article_text))
        
        articles = all_articles
        print(f"Found {len(articles)} articles by scanning")
    
    if len(articles) == 0:
        # Last resort: manually look for the article section
        print("\nSearching for 'Article 1' as a reference point...")
        article_1_pos = text.find('Article 1')
        if article_1_pos != -1:
            print(f"Found 'Article 1' at position {article_1_pos}")
            print("Text around Article 1:")
            print(text[article_1_pos:article_1_pos + 200])
    
    # Save articles
    saved = 0
    for article_num, article_text in articles:
        if not article_text.strip():
            continue
            
        safe_num = article_num.replace('.', '_')
        filename = f"Article_{safe_num}.txt"
        filepath = os.path.join(output_folder, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(article_text)
        
        saved += 1
        if saved % 20 == 0:
            print(f"  Saved {saved} articles...")
    
    print(f"\n✅ Extracted {saved} articles to '{output_folder}/'")
    return articles

if __name__ == "__main__":
    pdf_path = "OJ_L_202401689_EN_TXT.pdf"
    
    print("=" * 70)
    print("EU AI Act Article Extractor - Debug Version")
    print("=" * 70)
    
    # Try the most robust method first
    print("\n[Method 1] Robust extraction...")
    articles = extract_articles_robust(pdf_path, "eu_ai_act_articles_v1")
    
    if not articles:
        print("\n[Method 1] Failed. Trying Method 2...")
        print("\n[Method 2] Line-by-line extraction...")
        articles = extract_articles_by_line(pdf_path, "eu_ai_act_articles_v2")
    
    if not articles:
        print("\n[Method 2] Failed. Trying Method 3...")
        print("\n[Method 3] Manual parse...")
        articles = manual_parse_articles(pdf_path, "eu_ai_act_articles_v3")
    
    print("\n" + "=" * 70)
    print("Extraction complete!")
    print("=" * 70)