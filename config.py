import re
from .vector_store import VectorStoreManager
from .generator import MedicalGenerator
from .config import Config

class MedicalRAGPipeline:
    def __init__(self):
        self.vector_store = VectorStoreManager()
        self.generator = None

    def run(self, query):
        if not self.vector_store.index:
            if not self.vector_store.load(): return "Error: Index not found."
        
        results = self.vector_store.search(query)
        if not results: return "No relevant information found."
        
        if self.generator is None: self.generator = MedicalGenerator()
        
        messages = self.generator.build_prompt(query, results)
        answer = self.generator.generate(messages)
        
        # استخراج الـ Citations وربطها بالـ Metadata
        cited = sorted({int(n) for n in re.findall(r"\[S(\d+)\]", answer)})
        valid = [n for n in cited if 1 <= n <= len(results)]
        
        if valid:
            answer += "\n\n### Sources\n"
            for n in valid:
                res = results[n-1]
                meta = res["metadata"]
                answer += f"- [S{n}] Section {meta['section_number']}: {meta['section_title']}; pages {meta['pages']}\n"
        return answer
