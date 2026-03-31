import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

class AIActGuardrailGenerator:
    """Enhanced guardrail generator for EU AI Act with accurate article extraction"""
    
    def __init__(self):
        # Article-specific configurations
        self.article_configs = {
            '5': self._process_article_5,
            '6': self._process_article_6,
            '7': self._process_article_7,
            '9': self._process_article_9,
            '52': self._process_article_52,
            '95': self._process_article_95,
        }
        
        # Common patterns
        self.prohibition_patterns = {
            'explicit': [
                r'shall not be (?:placed on the market|put into service|used)',
                r'is prohibited',
                r'are prohibited',
                r'shall be prohibited',
                r'forbidden',
                r'shall not be allowed',
                r'may not be used',
                r'no person shall',
            ],
            'conditional': [
                r'shall be considered prohibited',
                r'constitutes a prohibited',
                r'falls within the prohibition',
            ]
        }
        
        self.penalty_patterns = [
            (r'up to (?:EUR|€)\s*(\d+(?:[.,]\d+)?)\s*(?:million)?', 'up to EUR {} million'),
            (r'(\d+)\s*%\s*of\s*(?:their\s*)?(?:total\s*)?(?:worldwide\s*)?annual\s*turnover', '{}% of global annual turnover'),
            (r'(?:EUR|€)\s*(\d+(?:[.,]\d+)?)\s*(?:million)?', 'EUR {} million'),
        ]
    
    def read_articles(self, folder_path: str) -> Dict[str, str]:
        """Read all article files from folder"""
        articles = {}
        folder = Path(folder_path)
        
        if not folder.exists():
            print(f"❌ Folder not found: {folder_path}")
            return articles
        
        article_files = sorted(folder.glob("Article_*.txt"))
        print(f"📁 Found {len(article_files)} article files")
        
        for file_path in article_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    article_num = self._extract_article_number(file_path.stem)
                    articles[article_num] = content
            except Exception as e:
                print(f"⚠️ Error reading {file_path.name}: {e}")
        
        return articles
    
    def _extract_article_number(self, filename: str) -> str:
        """Extract article number from filename"""
        match = re.search(r'Article_(\d+(?:_\d+)?)', filename)
        if match:
            return match.group(1).replace('_', '.')
        return filename
    
    def _process_article_5(self, text: str, article_num: str) -> Dict:
        """Specialized processing for Article 5 - Prohibited AI Practices"""
        
        # Split into paragraphs for better context
        paragraphs = self._split_paragraphs(text)
        
        prohibitions = []
        exceptions = []
        derogations = []
        
        # Article 5(1) - Main prohibitions
        prohibition_mapping = {
            'subliminal': "Harmful manipulation using subliminal techniques or purposefully manipulative/deceptive methods",
            'vulnerability': "Exploitation of vulnerabilities due to age, disability, or socio-economic situation",
            'social scoring': "Social scoring by public authorities leading to detrimental/unfavorable treatment",
            'predictive policing': "Individual risk assessment for predicting criminal offenses based on profiling",
            'facial scraping': "Untargeted scraping of facial images from internet/CCTV to create facial recognition databases",
            'emotion recognition': "Emotion recognition in workplaces and educational institutions (except medical/safety reasons)",
            'biometric categorization': "Biometric categorization inferring sensitive characteristics (race, politics, religion, etc.)",
            'rbi': "Real-time remote biometric identification in publicly accessible spaces for law enforcement"
        }
        
        # Check each prohibition
        text_lower = text.lower()
        for key, description in prohibition_mapping.items():
            if key in text_lower:
                prohibitions.append(description)
        
        # Article 5(2) - Exceptions for RBI
        if 'remote biometric identification' in text_lower and 'exhaustively listed' in text_lower:
            exceptions = [
                "Targeted search for potential victims of crime (including missing children)",
                "Prevention of specific, substantial, and imminent terrorist threat",
                "Localization or identification of perpetrator/suspect of serious crimes"
            ]
            derogations = [
                "Requires judicial or administrative authorization (except for urgent cases)",
                "Subject to appropriate time, geographic, and personal safeguards",
                "Must respect fundamental rights and be proportionate"
            ]
        
        # Article 5(3) - Not affecting other prohibitions
        if 'shall not affect' in text_lower and 'other prohibitions' in text_lower:
            prohibitions.append("Note: Additional prohibitions may apply under other EU or national laws")
        
        # Penalties reference (Article 99)
        penalties = [
            "Up to EUR 35,000,000 or 7% of global annual turnover for non-compliance"
        ]
        
        # Who it applies to
        applies_to = [
            "All providers placing AI systems on EU market",
            "All deployers using AI systems within EU",
            "Importers, distributors, and authorized representatives"
        ]
        
        # Keywords for matching
        keywords = [
            "prohibited", "biometric", "remote biometric identification", "rbi",
            "social scoring", "emotion recognition", "predictive policing",
            "manipulation", "vulnerabilities", "fundamental rights"
        ]
        
        return {
            'title': self._extract_title(text),
            'prohibits': prohibitions,
            'exceptions': exceptions,
            'derogations': derogations,
            'penalties': penalties,
            'applies_to': applies_to,
            'keywords': keywords,
            'source': f"Article {article_num} of Regulation (EU) 2024/1689",
            'applicability_date': "February 2, 2025",
            'enforcement_date': "August 2, 2025"
        }
    
    def _process_article_6(self, text: str, article_num: str) -> Dict:
        """Process Article 6 - Classification rules for high-risk AI systems"""
        
        text_lower = text.lower()
        
        high_risk_criteria = []
        if 'product safety' in text_lower or 'annex i' in text_lower:
            high_risk_criteria.append("AI systems used as safety components of products covered by EU harmonization legislation")
        
        if 'annex iii' in text_lower:
            high_risk_criteria.extend([
                "Biometric identification and categorization",
                "Critical infrastructure management",
                "Education and vocational training",
                "Employment and worker management",
                "Access to essential services",
                "Law enforcement",
                "Migration and border control",
                "Administration of justice",
                "Insurance and banking"
            ])
        
        return {
            'title': self._extract_title(text),
            'high_risk_criteria': high_risk_criteria,
            'applies_to': ['Providers of high-risk AI systems'],
            'keywords': ['high-risk', 'classification', 'annex', 'safety component'],
            'requires': [
                'Conformity assessment before placing on market',
                'Registration in EU database',
                'Compliance with requirements in Chapter 2'
            ]
        }
    
    def _process_article_52(self, text: str, article_num: str) -> Dict:
        """Process Article 52 - Transparency obligations"""
        
        text_lower = text.lower()
        
        transparency_obligations = []
        
        if 'emotion recognition' in text_lower or 'biometric' in text_lower:
            transparency_obligations.append(
                "Inform users when interacting with emotion recognition or biometric categorization systems"
            )
        
        if 'deep fake' in text_lower or 'synthetic' in text_lower:
            transparency_obligations.append(
                "Disclose that content is AI-generated or manipulated (deep fakes)"
            )
        
        if 'chatbot' in text_lower or 'conversational' in text_lower:
            transparency_obligations.append(
                "Inform users they are interacting with an AI system (chatbots, etc.)"
            )
        
        return {
            'title': self._extract_title(text),
            'requires': transparency_obligations,
            'applies_to': ['Providers and deployers of AI systems with user interaction'],
            'exceptions': ['Law enforcement activities', 'Artistic or creative works with safeguards'],
            'keywords': ['transparency', 'deep fake', 'synthetic', 'chatbot', 'emotion recognition']
        }
    
    def _process_article_95(self, text: str, article_num: str) -> Dict:
        """Process Article 95 - Penalties"""
        
        penalties = []
        
        # Extract penalties
        penalty_text = re.search(r'penalties?\s+(?:shall be|are|include).*?(?=Article|$)', text, re.IGNORECASE | re.DOTALL)
        if penalty_text:
            # Article 95 references Article 99
            penalties = [
                "Non-compliance with prohibited AI (Art.5): Up to EUR 35,000,000 or 7% of global turnover",
                "Non-compliance with high-risk AI requirements: Up to EUR 15,000,000 or 3% of global turnover",
                "Providing incorrect information: Up to EUR 7,500,000 or 1% of global turnover",
                "For SMEs: Lower amounts or percentage of turnover"
            ]
        
        return {
            'title': self._extract_title(text),
            'penalties': penalties,
            'applies_to': ['All entities subject to AI Act obligations'],
            'keywords': ['penalties', 'fines', 'sanctions', 'turnover', 'enforcement']
        }
    
    def _process_article_7(self, text: str, article_num: str) -> Dict:
        """Process Article 7 - Amendments to high-risk classification"""
        
        return {
            'title': self._extract_title(text),
            'requires': [
                'Commission empowered to amend Annex III through delegated acts',
                'Must consider adverse impact on fundamental rights',
                'Consultation with relevant stakeholders'
            ],
            'keywords': ['amendment', 'high-risk', 'delegated act', 'annex iii']
        }
    
    def _process_article_9(self, text: str, article_num: str) -> Dict:
        """Process Article 9 - Risk management system"""
        
        return {
            'title': self._extract_title(text),
            'requires': [
                'Establish and maintain risk management system throughout AI system lifecycle',
                'Identify and analyze known and foreseeable risks',
                'Estimate and evaluate risks (including reasonably foreseeable misuse)',
                'Implement appropriate risk management measures',
                'Testing to ensure critical performance'
            ],
            'keywords': ['risk management', 'risk assessment', 'testing', 'lifecycle']
        }
    
    def _split_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs for processing"""
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _extract_title(self, text: str) -> str:
        """Extract article title"""
        lines = text.strip().split('\n')
        if lines:
            # Remove article number from title
            title = re.sub(r'^Article\s+\d+(?:\.\d+)?\s*', '', lines[0], flags=re.IGNORECASE)
            return title.strip()[:100]
        return ""
    
    def _extract_general_requirements(self, text: str, article_num: str) -> Dict:
        """General extraction for articles without specialized handlers"""
        
        text_lower = text.lower()
        sentences = re.split(r'[.!?]+', text)
        
        requirements = []
        prohibitions = []
        exceptions = []
        
        # Extract requirements
        requirement_triggers = [
            'shall ensure', 'shall establish', 'shall implement', 'shall provide',
            'shall be designed', 'shall be developed', 'shall maintain', 'shall keep',
            'shall document', 'shall be subject to', 'shall be capable of'
        ]
        
        for sentence in sentences[:15]:  # Limit to first 15 sentences
            sentence = sentence.strip()
            if not sentence or len(sentence) > 200:
                continue
            
            for trigger in requirement_triggers:
                if trigger in sentence.lower():
                    # Clean up the requirement
                    clean = re.sub(r'Article\s+\d+(?:\.\d+)?', '', sentence)
                    clean = ' '.join(clean.split())
                    if len(clean) > 120:
                        clean = clean[:117] + "..."
                    if clean not in requirements:
                        requirements.append(clean)
                    break
        
        # Extract prohibitions
        for sentence in sentences[:15]:
            sentence = sentence.strip()
            if not sentence or len(sentence) > 200:
                continue
            
            for pattern_list in self.prohibition_patterns.values():
                for pattern in pattern_list:
                    if re.search(pattern, sentence.lower()):
                        clean = re.sub(r'Article\s+\d+(?:\.\d+)?', '', sentence)
                        clean = ' '.join(clean.split())
                        if len(clean) > 120:
                            clean = clean[:117] + "..."
                        if clean not in prohibitions:
                            prohibitions.append(clean)
                        break
        
        # Extract exceptions
        exception_triggers = ['except', 'unless', 'does not apply', 'derogation']
        for sentence in sentences[:10]:
            for trigger in exception_triggers:
                if trigger in sentence.lower():
                    clean = sentence.strip()[:100]
                    if clean not in exceptions:
                        exceptions.append(clean)
        
        return {
            'title': self._extract_title(text),
            'prohibits': prohibitions[:5],
            'requires': requirements[:8],
            'exceptions': exceptions[:3],
            'keywords': self._extract_keywords(text)
        }
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract key terms from text"""
        important_terms = [
            'biometric', 'high-risk', 'transparency', 'fundamental rights',
            'conformity', 'notified body', 'quality management', 'risk assessment',
            'human oversight', 'data governance', 'post-market monitoring',
            'serious incident', 'market surveillance', 'prohibited',
            'remote biometric identification', 'emotion recognition',
            'social scoring', 'profiling', 'general-purpose AI',
            'foundation model', 'generative AI', 'safety', 'security'
        ]
        
        found_terms = []
        text_lower = text.lower()
        
        for term in important_terms:
            if term in text_lower:
                found_terms.append(term)
        
        return found_terms[:10]
    
    def process_article(self, content: str, article_num: str) -> Dict:
        """Process an article with specialized handler if available"""
        
        # Use specialized handler for key articles
        base_article = article_num.split('.')[0]  # Handle sub-articles
        if base_article in self.article_configs:
            return self.article_configs[base_article](content, article_num)
        
        # Fall back to general extraction
        return self._extract_general_requirements(content, article_num)
    
    def generate_guardrail(self, article_data: Dict, article_num: str) -> str:
        """Generate compact guardrail text for an article"""
        
        output = f"""[ART{article_num}] {article_data.get('title', '')[:60]}

"""
        
        # Prohibitions (Article 5 and others)
        if article_data.get('prohibits'):
            output += "🚫 PROHIBITS:\n"
            for p in article_data['prohibits'][:8]:
                output += f"   • {p}\n"
            output += "\n"
        
        # Requirements
        if article_data.get('requires'):
            output += "✅ REQUIRES:\n"
            for r in article_data['requires'][:8]:
                output += f"   • {r}\n"
            output += "\n"
        
        # High-risk criteria (Article 6)
        if article_data.get('high_risk_criteria'):
            output += "⚠️ HIGH-RISK CRITERIA:\n"
            for c in article_data['high_risk_criteria'][:10]:
                output += f"   • {c}\n"
            output += "\n"
        
        # Exceptions (important for Article 5)
        if article_data.get('exceptions'):
            output += "🔓 EXCEPTIONS:\n"
            for e in article_data['exceptions'][:5]:
                output += f"   • {e}\n"
            output += "\n"
        
        # Derogations (Article 5 specific)
        if article_data.get('derogations'):
            output += "⚖️ DEROGATIONS:\n"
            for d in article_data['derogations'][:4]:
                output += f"   • {d}\n"
            output += "\n"
        
        # Who it applies to
        if article_data.get('applies_to'):
            output += "👥 APPLIES TO:\n"
            for a in article_data['applies_to'][:5]:
                output += f"   • {a}\n"
            output += "\n"
        
        # Penalties
        if article_data.get('penalties'):
            output += "💰 PENALTIES:\n"
            for p in article_data['penalties'][:3]:
                output += f"   • {p}\n"
            output += "\n"
        
        # Keywords
        if article_data.get('keywords'):
            output += "🔑 KEYWORDS: " + ", ".join(article_data['keywords']) + "\n\n"
        
        # Metadata
        if article_data.get('source'):
            output += f"📚 SOURCE: {article_data['source']}\n"
        
        if article_data.get('applicability_date'):
            output += f"📅 APPLICABLE FROM: {article_data['applicability_date']}\n"
        
        output += "-" * 70 + "\n"
        
        return output
    
    def generate_all_guardrails(self, articles_folder: str, output_folder: str):
        """Generate all guardrails"""
        
        os.makedirs(output_folder, exist_ok=True)
        
        print("=" * 70)
        print("EU AI ACT - ACCURATE GUARDRAIL GENERATOR")
        print("Based on Regulation (EU) 2024/1689")
        print("=" * 70)
        
        # Read articles
        articles = self.read_articles(articles_folder)
        
        if not articles:
            print("\n❌ No articles found!")
            return
        
        all_guardrails = {}
        processed_data = {}
        
        # Process each article
        for article_num, content in articles.items():
            print(f"📝 Processing Article {article_num}...", end=" ")
            
            # Process article content
            article_data = self.process_article(content, article_num)
            processed_data[article_num] = article_data
            
            # Generate guardrail text
            guardrail_text = self.generate_guardrail(article_data, article_num)
            all_guardrails[article_num] = guardrail_text
            
            # Save individual file
            output_file = os.path.join(output_folder, f"ART_{article_num.replace('.', '_')}_guardrail.txt")
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(guardrail_text)
            
            print("✓")
        
        # Create master file
        master_path = os.path.join(output_folder, "00_ALL_GUARDRAILS.txt")
        with open(master_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("EU AI ACT - COMPREHENSIVE GUARDRAILS\n")
            f.write("Regulation (EU) 2024/1689 - Full Text Analysis\n")
            f.write("=" * 70 + "\n\n")
            
            for article_num in sorted(all_guardrails.keys(), key=lambda x: float(x.split('.')[0])):
                f.write(all_guardrails[article_num])
        
        # Save JSON data
        json_path = os.path.join(output_folder, "guardrails_complete.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False)
        
        # Create search index
        search_index = self._create_search_index(processed_data)
        index_path = os.path.join(output_folder, "search_index.json")
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(search_index, f, indent=2, ensure_ascii=False)
        
        # Create matcher module
        self._create_matcher_module(processed_data, search_index, output_folder)
        
        # Summary
        print("\n" + "=" * 70)
        print("✅ GUARDRAILS GENERATED SUCCESSFULLY!")
        print("=" * 70)
        print(f"\n📁 Output folder: {output_folder}/")
        print(f"📄 Files created:")
        print(f"   • 00_ALL_GUARDRAILS.txt - Complete guardrails in text format")
        print(f"   • guardrails_complete.json - Structured data for programmatic use")
        print(f"   • search_index.json - Keyword-based search index")
        print(f"   • guardrail_matcher.py - Python module for compliance checking")
        print(f"   • ART_*_guardrail.txt - Individual article guardrails")
        
        # Article 5 summary
        if '5' in processed_data:
            print("\n" + "=" * 70)
            print("📋 ARTICLE 5 - PROHIBITED AI PRACTICES")
            print("=" * 70)
            art5 = processed_data['5']
            print(f"\n🚫 {len(art5.get('prohibits', []))} Prohibitions identified")
            for p in art5.get('prohibits', [])[:5]:
                print(f"   • {p}")
            print(f"\n🔓 {len(art5.get('exceptions', []))} Exceptions")
            print(f"💰 Penalties: {art5.get('penalties', ['Not specified'])[0]}")
        
        print("\n" + "=" * 70)
    
    def _create_search_index(self, processed_data: Dict) -> Dict:
        """Create search index for quick lookup"""
        
        search_index = {}
        
        for article_num, data in processed_data.items():
            # Index keywords
            for keyword in data.get('keywords', []):
                if keyword not in search_index:
                    search_index[keyword] = []
                if article_num not in search_index[keyword]:
                    search_index[keyword].append(article_num)
            
            # Index prohibitions
            for prohibition in data.get('prohibits', []):
                trigger = ' '.join(prohibition.split()[:4]).lower()
                if trigger not in search_index:
                    search_index[trigger] = []
                if article_num not in search_index[trigger]:
                    search_index[trigger].append(article_num)
        
        return search_index
    
    def _create_matcher_module(self, processed_data: Dict, search_index: Dict, output_folder: str):
        """Create Python matcher module"""
        
        matcher_code = f'''# Auto-generated guardrail matcher for EU AI Act
# Based on Regulation (EU) 2024/1689

import re
import json
from typing import List, Dict, Optional, Tuple

# Load guardrail data
GUARDRAILS = {json.dumps(processed_data, indent=2, ensure_ascii=False)}

# Search index
SEARCH_INDEX = {json.dumps(search_index, indent=2, ensure_ascii=False)}

class AIActGuardrail:
    """Guardrail matcher for EU AI Act compliance"""
    
    @staticmethod
    def check_violation(query: str) -> Optional[Dict]:
        """
        Check if a query would violate any prohibition
        Returns violation details if found
        """
        query_lower = query.lower()
        
        # Priority articles for prohibitions
        priority_articles = ['5', '52']  # Article 5 and transparency
        
        for article_num in priority_articles:
            if article_num not in GUARDRAILS:
                continue
            
            data = GUARDRAILS[article_num]
            for prohibition in data.get('prohibits', []):
                # Check for key terms in prohibition
                prohibition_keywords = set(re.findall(r'\\b\\w+\\b', prohibition.lower()))
                query_keywords = set(re.findall(r'\\b\\w+\\b', query_lower))
                
                # Require at least 2 significant keyword matches
                significant_matches = len(prohibition_keywords & query_keywords)
                if significant_matches >= 2:
                    return {{
                        'article': article_num,
                        'title': data.get('title', ''),
                        'violation': prohibition,
                        'penalty': data.get('penalties', ['See Article 95'])[0],
                        'exceptions': data.get('exceptions', [])[:2]
                    }}
        
        return None
    
    @staticmethod
    def check_requirements(query: str) -> List[Dict]:
        """
        Check requirements that apply to a use case
        Returns list of relevant requirements
        """
        query_lower = query.lower()
        relevant = []
        
        for article_num, data in GUARDRAILS.items():
            # Skip if no requirements
            if not data.get('requires'):
                continue
            
            score = 0
            matched_requirements = []
            
            for req in data['requires']:
                req_keywords = set(re.findall(r'\\b\\w+\\b', req.lower()))
                query_keywords = set(re.findall(r'\\b\\w+\\b', query_lower))
                
                if len(req_keywords & query_keywords) >= 2:
                    matched_requirements.append(req)
                    score += 3
            
            if score > 0:
                relevant.append({{
                    'article': article_num,
                    'title': data.get('title', ''),
                    'requirements': matched_requirements[:3],
                    'score': score
                }})
        
        return sorted(relevant, key=lambda x: x['score'], reverse=True)[:5]
    
    @staticmethod
    def search(query: str) -> List[Dict]:
        """
        Search for relevant articles
        """
        query_lower = query.lower()
        results = {{}}
        
        # Check keyword matches
        for keyword, articles in SEARCH_INDEX.items():
            if keyword in query_lower:
                for article in articles:
                    results[article] = results.get(article, 0) + 3
        
        # Add from prohibitions and requirements
        for article_num, data in GUARDRAILS.items():
            if article_num in results:
                continue
            
            score = 0
            for text in data.get('prohibits', []) + data.get('requires', []):
                if any(word in query_lower for word in text.lower().split()[:4]):
                    score += 1
            
            if score >= 2:
                results[article_num] = score
        
        # Format results
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        
        return [{{
            'article': art,
            'title': GUARDRAILS[art].get('title', ''),
            'keywords': GUARDRAILS[art].get('keywords', [])[:5],
            'score': score
        }} for art, score in sorted_results[:5]]

# Quick test
if __name__ == "__main__":
    guardrail = AIActGuardrail()
    
    test_queries = [
        "Can I use facial recognition to identify people in public spaces?",
        "I want to build an AI system that evaluates employee emotions",
        "What are the requirements for a high-risk AI system?",
        "How much is the fine for using prohibited AI practices?"
    ]
    
    for query in test_queries:
        print(f"\\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)
        
        violation = guardrail.check_violation(query)
        if violation:
            print(f"⚠️ VIOLATION DETECTED!")
            print(f"   Article {violation['article']}: {violation['title'][:60]}")
            print(f"   Prohibition: {violation['violation']}")
            print(f"   Penalty: {violation['penalty']}")
            if violation.get('exceptions'):
                print(f"   Exceptions: {', '.join(violation['exceptions'][:2])}")
        else:
            requirements = guardrail.check_requirements(query)
            if requirements:
                print(f"📋 APPLICABLE REQUIREMENTS:")
                for req in requirements[:3]:
                    print(f"   • Article {req['article']}: {req['title'][:60]}")
                    for r in req['requirements'][:2]:
                        print(f"     - {r[:80]}")
'''
        
        matcher_path = os.path.join(output_folder, "guardrail_matcher.py")
        with open(matcher_path, 'w', encoding='utf-8') as f:
            f.write(matcher_code)
        
        print(f"✅ Created matcher module: {matcher_path}")


def main():
    """Main execution"""
    
    # Configuration
    articles_folder = "eu_ai_act_articles"
    output_folder = "ai_act_guardrails"
    
    # Generate guardrails
    generator = AIActGuardrailGenerator()
    generator.generate_all_guardrails(articles_folder, output_folder)


if __name__ == "__main__":
    main()