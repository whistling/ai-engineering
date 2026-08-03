#!/usr/bin/env python3
"""
LLM Wiki Demo - main.py
Provides a CLI tool to ingest, query, and lint a local Markdown wiki using LLMs.
Author: Antigravity Agent
"""

import os
import re
import json
import urllib.request
import urllib.error
import argparse
from datetime import datetime
from pathlib import Path

# ==========================================
# LLM Provider Clients
# ==========================================

class LLMClient:
    def __init__(self, provider=None, model=None, api_key=None, endpoint=None):
        self.provider = provider or os.environ.get("LLM_PROVIDER", "ollama").lower()
        self.model = model
        self.api_key = api_key
        self.endpoint = endpoint
        
        # Configure defaults based on provider
        if self.provider == "gemini":
            self.api_key = self.api_key or os.environ.get("GEMINI_API_KEY")
            self.model = self.model or "gemini-2.5-flash"
            self.endpoint = self.endpoint or f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        elif self.provider == "openai":
            self.api_key = self.api_key or os.environ.get("OPENAI_API_KEY")
            self.model = self.model or "gpt-4o-mini"
            self.endpoint = self.endpoint or "https://api.openai.com/v1/chat/completions"
        else: # ollama
            self.provider = "ollama"
            self.model = self.model or "qwen2.5:7b"  # Common local model
            self.endpoint = self.endpoint or "http://localhost:11434/api/chat"

    def generate(self, system_prompt, user_prompt, response_json=False):
        """Generates content from the LLM, optionally requesting JSON output."""
        if self.provider == "gemini":
            return self._call_gemini(system_prompt, user_prompt, response_json)
        elif self.provider == "openai":
            return self._call_openai(system_prompt, user_prompt, response_json)
        else:
            return self._call_ollama(system_prompt, user_prompt, response_json)

    def _call_ollama(self, system, user, response_json):
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            "stream": False,
            "options": {
                "temperature": 0.2
            }
        }
        if response_json:
            payload["format"] = "json"

        headers = {"Content-Type": "application/json"}
        req = urllib.request.Request(self.endpoint, data=json.dumps(payload).encode("utf-8"), headers=headers)
        
        try:
            with urllib.request.urlopen(req) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                return res_data["message"]["content"]
        except Exception as e:
            print(f"[Ollama Error] Ensure Ollama is running at {self.endpoint} and model '{self.model}' is pulled.")
            raise e

    def _call_gemini(self, system, user, response_json):
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required for Gemini provider.")
        
        url = f"{self.endpoint}?key={self.api_key}"
        
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": f"System Instruction: {system}\n\nUser Input:\n{user}"}]
                }
            ],
            "generationConfig": {
                "temperature": 0.2
            }
        }
        if response_json:
            payload["generationConfig"]["responseMimeType"] = "application/json"

        headers = {"Content-Type": "application/json"}
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
        
        try:
            with urllib.request.urlopen(req) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                text = res_data["candidates"][0]["content"]["parts"][0]["text"]
                return text
        except urllib.error.HTTPError as e:
            print(f"[Gemini API Error] {e.read().decode('utf-8')}")
            raise e

    def _call_openai(self, system, user, response_json):
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required for OpenAI provider.")
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            "temperature": 0.2
        }
        if response_json:
            payload["response_format"] = {"type": "json_object"}

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        req = urllib.request.Request(self.endpoint, data=json.dumps(payload).encode("utf-8"), headers=headers)
        
        try:
            with urllib.request.urlopen(req) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                return res_data["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            print(f"[OpenAI API Error] {e.read().decode('utf-8')}")
            raise e

# ==========================================
# Wiki Helper Functions
# ==========================================

def slugify(text):
    """Converts a title or entity name to a safe filename slug."""
    text = text.strip().lower()
    text = re.sub(r'[\s_]+', '_', text)
    text = re.sub(r'[^\w\-]', '', text)
    return text

def parse_frontmatter(content):
    """Extracts frontmatter dict and remaining content."""
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            fm_text = parts[1]
            body = parts[2]
            fm = {}
            for line in fm_text.splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    k = k.strip()
                    v = v.strip()
                    # Basic list parsing
                    if v.startswith("[") and v.endswith("]"):
                        v = [item.strip().strip('"').strip("'") for item in v[1:-1].split(",") if item.strip()]
                    elif v.startswith("-"):
                        # Handled if we parse line-by-line, keeping it simple here
                        pass
                    fm[k] = v
            return fm, body
    return {}, content

# ==========================================
# Wiki Operations Class
# ==========================================

class LLMWiki:
    def __init__(self, base_dir, llm_client):
        self.base_dir = Path(base_dir)
        self.raw_dir = self.base_dir / "raw"
        self.wiki_dir = self.base_dir / "wiki"
        self.llm = llm_client
        
        # Ensure directories exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.wiki_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Special Files
        self._init_special_files()

    def _init_special_files(self):
        index_file = self.wiki_dir / "index.md"
        if not index_file.exists():
            index_file.write_text(
                "---\ntitle: Wiki Index\ntype: system\nlast_updated: " + 
                datetime.now().strftime("%Y-%m-%d") + "\n---\n"
                "# Wiki Index\n\nWelcome to your LLM-maintained Wiki.\n\n## Pages\n*No pages ingested yet.*\n",
                encoding="utf-8"
            )
            
        log_file = self.wiki_dir / "log.md"
        if not log_file.exists():
            log_file.write_text(
                "---\ntitle: Wiki Activity Log\ntype: system\n---\n"
                "# Wiki Activity Log\n\nChronological record of wiki operations.\n\n",
                encoding="utf-8"
            )

    def get_ingested_files(self):
        """Reads log.md to find already ingested raw files."""
        log_file = self.wiki_dir / "log.md"
        content = log_file.read_text(encoding="utf-8")
        # Match log patterns: ## [2026-07-04] Ingested | filename.txt
        ingested = re.findall(r"Ingested\s*\|\s*([^\n\r]+)", content)
        return [f.strip() for f in ingested]

    def log_operation(self, op_type, detail):
        """Appends an entry to log.md."""
        log_file = self.wiki_dir / "log.md"
        date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"## [{date_str}] {op_type} | {detail}\n\n"
        with log_file.open("a", encoding="utf-8") as f:
            f.write(entry)

    def update_index(self):
        """Updates index.md by scanning the wiki directory."""
        pages = []
        for file in self.wiki_dir.glob("*.md"):
            if file.name in ["index.md", "log.md"]:
                continue
            content = file.read_text(encoding="utf-8")
            fm, _ = parse_frontmatter(content)
            title = fm.get("title", file.stem.replace("_", " ").title())
            p_type = fm.get("type", "General")
            pages.append((title, file.name, p_type))
            
        pages.sort(key=lambda x: (x[2], x[0])) # Sort by type, then title
        
        index_content = [
            "---",
            "title: Wiki Index",
            "type: system",
            f"last_updated: {datetime.now().strftime('%Y-%m-%d')}",
            "---",
            "# Wiki Index",
            "\nWelcome to your LLM-maintained Wiki.",
            "\n## Categories\n"
        ]
        
        current_type = None
        for title, fname, p_type in pages:
            if p_type != current_type:
                current_type = p_type
                index_content.append(f"\n### {current_type}")
            index_content.append(f"- [[{fname[:-3]}|{title}]]")
            
        (self.wiki_dir / "index.md").write_text("\n".join(index_content) + "\n", encoding="utf-8")

    def ingest_file(self, raw_file_path):
        """Processes a single raw file: extracts concepts, writes/updates wiki pages."""
        raw_path = Path(raw_file_path)
        print(f"\n[Ingest] Processing: {raw_path.name}")
        content = raw_path.read_text(encoding="utf-8")
        
        system_prompt = (
            "You are an expert knowledge engineer and wiki maintainer.\n"
            "Analyze the source text provided by the user. Extract key entities, projects, concepts, or terms.\n"
            "For each key concept or entity, you will provide a structured, high-fidelity synthesis.\n"
            "You MUST preserve technical details: include concrete prompt templates, code blocks, parameter details, mathematical formulas, raw prompt configurations, and specific examples verbatim from the text. DO NOT write dry or generic summaries.\n"
            "Respond ONLY with a valid JSON object matching this schema:\n"
            "{\n"
            "  \"source_summary\": \"A concise 2-3 sentence overview of this source document.\",\n"
            "  \"entities\": [\n"
            "    {\n"
            "      \"name\": \"Name of the concept/entity (e.g. RAG, Ollama, John Doe)\",\n"
            "      \"type\": \"One of: Concept, Technology, Person, Project, Standard, Organization\",\n"
            "      \"description\": \"A highly detailed and structured explanation (in Markdown format). "
            "You MUST include: \\n1. Core Mechanism: Detailed technical explanation. \\n2. Examples/Templates: Specific examples, prompt templates, or code snippets from the text. \\n3. Parameters/Caveats: Key parameters (e.g., temperature), rules, edge cases, and limitations.\",\n"
            "      \"links\": [\"List of other concept/entity names mentioned in this context that should be interlinked\"]\n"
            "    }\n"
            "  ]\n"
            "}"
        )
        
        user_prompt = f"Source Document Name: {raw_path.name}\n\nContent:\n{content}"
        
        print(" -> Asking LLM to extract knowledge...")
        response_text = self.llm.generate(system_prompt, user_prompt, response_json=True)
        
        try:
            # Clean potential LLM markdown code blocks
            clean_json = response_text.strip()
            if clean_json.startswith("```json"):
                clean_json = clean_json[7:]
            if clean_json.endswith("```"):
                clean_json = clean_json[:-3]
            clean_json = clean_json.strip()
            
            data = json.loads(clean_json)
        except Exception as e:
            print(f"[Extraction Error] Failed to parse JSON response: {response_text}")
            raise e

        # Write or update pages for each entity
        for entity in data.get("entities", []):
            name = entity["name"]
            slug = slugify(name)
            wiki_file = self.wiki_dir / f"{slug}.md"
            
            # Format YAML list of sources
            source_entry = raw_path.name
            
            if wiki_file.exists():
                print(f" -> Updating existing wiki page: [[{slug}]] ({name})")
                existing_content = wiki_file.read_text(encoding="utf-8")
                
                merge_system = (
                    "You are a wiki editor. Merge the NEW details into the EXISTING wiki article.\n"
                    "Do NOT drop, summarize away, or compromise on specific code snippets, prompt templates, parameters, formulas, or practical examples.\n"
                    "If both old and new texts contain different prompt templates or examples, preserve BOTH under separate sub-headings (e.g., Example 1, Example 2).\n"
                    "Do not duplicate sections. Update outdated facts and expand summaries smoothly.\n"
                    "Maintain the frontmatter structure. Append the new source name to the list of 'sources'.\n"
                    "Output ONLY the final merged Markdown file content starting with `---`."
                )
                merge_user = (
                    f"EXISTING ARTICLE:\n{existing_content}\n\n"
                    f"NEW DETAILS TO MERGE:\n"
                    f"Title: {name}\nDescription: {entity['description']}\n"
                    f"New Source: {source_entry}\n"
                    f"Suggested Links: {', '.join(entity.get('links', []))}"
                )
                merged_content = self.llm.generate(merge_system, merge_user, response_json=False)
                # Strip markdown fence if LLM added it
                if merged_content.strip().startswith("```markdown"):
                    merged_content = merged_content.strip()[11:]
                if merged_content.strip().startswith("```"):
                    merged_content = merged_content.strip()[3:]
                if merged_content.strip().endswith("```"):
                    merged_content = merged_content.strip()[:-3]
                
                wiki_file.write_text(merged_content.strip() + "\n", encoding="utf-8")
            else:
                print(f" -> Creating new wiki page: [[{slug}]] ({name})")
                # Format references and outgoing links
                out_links = "\n".join([f"- [[{slugify(l)}|{l}]]" for l in entity.get("links", []) if l.strip().lower() != name.lower()])
                
                new_content = (
                    "---\n"
                    f"title: {name}\n"
                    f"type: {entity['type']}\n"
                    f"last_updated: {datetime.now().strftime('%Y-%m-%d')}\n"
                    f"sources:\n"
                    f"  - \"{source_entry}\"\n"
                    "---\n"
                    f"# {name}\n\n"
                    f"{entity['description']}\n\n"
                    "## Outgoing Links\n"
                    f"{out_links if out_links else '*None*'}\n\n"
                    "## References\n"
                    "- [[index|Wiki Index]]\n"
                )
                wiki_file.write_text(new_content, encoding="utf-8")

        # Create source summary page
        src_slug = f"src_{slugify(raw_path.name)}"
        src_file = self.wiki_dir / f"{src_slug}.md"
        src_content = (
            "---\n"
            f"title: \"Source: {raw_path.name}\"\n"
            "type: Source Summary\n"
            f"last_updated: {datetime.now().strftime('%Y-%m-%d')}\n"
            "---\n"
            f"# Source: {raw_path.name}\n\n"
            f"{data.get('source_summary', 'No summary generated.')}\n\n"
            "## Entities Extracted\n"
            + "\n".join([f"- [[{slugify(e['name'])}|{e['name']}]] ({e['type']})" for e in data.get("entities", [])]) + "\n\n"
            "## References\n"
            "- [[index|Wiki Index]]\n"
        )
        src_file.write_text(src_content, encoding="utf-8")
        
        # Log and Index
        self.log_operation("Ingested", raw_path.name)
        self.update_index()
        print(f"[Success] Ingested {raw_path.name} successfully!")

    def run_query(self, query_text):
        """Executes a query against the wiki by finding relevant files and letting the LLM answer."""
        print(f"\n[Query] Processing question: '{query_text}'")
        
        # Step 1: Read index to find what we have
        index_path = self.wiki_dir / "index.md"
        index_content = index_path.read_text(encoding="utf-8")
        
        # Step 2: Ask LLM which wiki pages are relevant to answer the query
        selection_system = (
            "You are a wiki librarian.\n"
            "Based on the wiki index and the user's query, return a JSON list of file names (without .md extension)\n"
            "that are relevant and should be read to answer the question.\n"
            "Respond ONLY with a JSON object: { \"relevant_pages\": [\"slug1\", \"slug2\"] }"
        )
        selection_user = f"Wiki Index:\n{index_content}\n\nUser Query: {query_text}"
        
        print(" -> Identifying relevant wiki pages...")
        res_selection = self.llm.generate(selection_system, selection_user, response_json=True)
        
        try:
            # Clean and parse JSON
            clean_json = res_selection.strip()
            if clean_json.startswith("```json"):
                clean_json = clean_json[7:]
            if clean_json.endswith("```"):
                clean_json = clean_json[:-3]
            clean_json = clean_json.strip()
            pages_to_read = json.loads(clean_json).get("relevant_pages", [])
        except Exception:
            pages_to_read = []
            
        print(f" -> Selected pages to inspect: {pages_to_read}")
        
        # Step 3: Read content of selected files
        context_parts = []
        for page in pages_to_read:
            filename = f"{slugify(page)}.md"
            page_path = self.wiki_dir / filename
            if page_path.exists():
                context_parts.append(f"--- File: {filename} ---\n{page_path.read_text(encoding='utf-8')}")
                
        # If no page matched, search fallback: read all summaries
        if not context_parts:
            print(" -> Fallback: scanning all source summary files...")
            for file in self.wiki_dir.glob("src_*.md"):
                context_parts.append(f"--- File: {file.name} ---\n{file.read_text(encoding='utf-8')}")
                
        context = "\n\n".join(context_parts)
        
        # Step 4: Ask LLM to answer using only the read pages
        answer_system = (
            "You are an expert researcher. Answer the user's query using ONLY the provided Wiki file contents.\n"
            "Cite the file names (e.g. [[slug]]) when referencing facts.\n"
            "If the answer cannot be found in the files, state what information is missing."
        )
        answer_user = f"WIKI CONTEXT:\n{context}\n\nQUERY: {query_text}"
        
        print(" -> Generating answer...")
        answer = self.llm.generate(answer_system, answer_user)
        
        # Log the query
        self.log_operation("Query", query_text)
        return answer

    def lint_wiki(self):
        """Runs heuristic audit checks over the wiki pages."""
        print("\n[Lint] Auditing wiki integrity...")
        
        pages = list(self.wiki_dir.glob("*.md"))
        all_slugs = {p.stem for p in pages}
        
        dangling_links = {}
        orphans = set(all_slugs) - {"index", "log"}
        missing_frontmatter = []
        
        for p in pages:
            if p.name in ["index.md", "log.md"]:
                continue
            content = p.read_text(encoding="utf-8")
            
            # Check frontmatter
            if not content.startswith("---"):
                missing_frontmatter.append(p.name)
            else:
                parts = content.split("---", 2)
                if len(parts) < 3:
                    missing_frontmatter.append(p.name)
            
            # Find markdown links: [[target]] or [[target|label]]
            links = re.findall(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]", content)
            for link in links:
                target_slug = slugify(link)
                # Mark target as not an orphan (someone links to it)
                if target_slug in orphans:
                    orphans.remove(target_slug)
                
                # Check if target exists
                if target_slug not in all_slugs and target_slug != "index":
                    if p.name not in dangling_links:
                        dangling_links[p.name] = []
                    dangling_links[p.name].append(link)
                    
        # Log lint pass
        self.log_operation("Lint", f"Run lint audit. Found {len(dangling_links)} files with dangling links, {len(orphans)} orphans.")
        
        return {
            "dangling_links": dangling_links,
            "orphans": list(orphans),
            "missing_frontmatter": missing_frontmatter
        }

# ==========================================
# CLI Controller
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="LLM Wiki - Local Second Brain Demo")
    parser.add_argument("action", choices=["ingest", "query", "lint", "status"], help="Action to perform")
    parser.add_argument("--query", "-q", help="The search query or question")
    parser.add_argument("--provider", default="ollama", choices=["ollama", "gemini", "openai"], help="LLM Provider")
    parser.add_argument("--model", help="LLM Model name")
    parser.add_argument("--endpoint", help="Ollama or API Endpoint URL")
    
    args = parser.parse_args()
    
    # Initialize LLM Client
    client = LLMClient(provider=args.provider, model=args.model, endpoint=args.endpoint)
    
    # Initialize Wiki
    wiki = LLMWiki(base_dir=os.getcwd(), llm_client=client)
    
    if args.action == "status":
        ingested = wiki.get_ingested_files()
        print("\n=== Wiki Status ===")
        print(f"Directory: {os.getcwd()}")
        print(f"Total Source files Ingested: {len(ingested)}")
        for f in ingested:
            print(f" - {f}")
        print("===================")
        
    elif args.action == "ingest":
        # Find raw files not yet ingested
        ingested = set(wiki.get_ingested_files())
        raw_files = [f for f in wiki.raw_dir.iterdir() if f.is_file() and not f.name.startswith(".")]
        
        to_ingest = [f for f in raw_files if f.name not in ingested]
        
        if not to_ingest:
            print("\n[Ingest] All raw files are already ingested! Drop new files in the 'raw/' folder.")
            return
            
        print(f"\n[Ingest] Found {len(to_ingest)} new files to ingest.")
        for file_path in to_ingest:
            try:
                wiki.ingest_file(file_path)
            except Exception as e:
                print(f"[Error] Failed to process {file_path.name}: {e}")
                
    elif args.action == "query":
        query_text = args.query
        if not query_text:
            query_text = input("\nEnter your query/question: ").strip()
        if not query_text:
            print("Query cannot be empty.")
            return
            
        try:
            answer = wiki.run_query(query_text)
            print("\n=== LLM Wiki Answer ===")
            print(answer)
            print("=======================")
        except Exception as e:
            print(f"[Error] Failed to execute query: {e}")
            
    elif args.action == "lint":
        report = wiki.lint_wiki()
        print("\n=== Wiki Lint Audit Report ===")
        
        print("\n1. Missing Frontmatter:")
        if report["missing_frontmatter"]:
            for f in report["missing_frontmatter"]:
                print(f" ❌ {f}")
        else:
            print("  Check: 🟢 All wiki pages have frontmatter.")
            
        print("\n2. Dangling Links (linked pages that don't exist):")
        if report["dangling_links"]:
            for src, targets in report["dangling_links"].items():
                print(f" ⚠️  In '{src}':")
                for t in targets:
                    print(f"    -> [[{t}]] (Not Found)")
        else:
            print("  Check: 🟢 No dangling links.")
            
        print("\n3. Orphan Pages (no other pages link to them):")
        if report["orphans"]:
            for o in report["orphans"]:
                print(f" ℹ️  [[{o}]]")
        else:
            print("  Check: 🟢 No orphan pages.")
        print("\n==============================")

if __name__ == "__main__":
    main()
