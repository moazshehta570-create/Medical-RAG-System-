import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from .config import Config

class MedicalGenerator:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(Config.LLM_MODEL_NAME, trust_remote_code=True)
        
        # إعدادات الـ 4-bit لتقليل استهلاك الذاكرة
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            Config.LLM_MODEL_NAME,
            quantization_config=quantization_config,
            device_map={"": 0} if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        self.model.eval()

    def build_prompt(self, query, results):
        context_parts = []
        for i, res in enumerate(results, start=1):
            pages = ", ".join(map(str, res["metadata"].get("pages", [])))
            context_parts.append(
                f"[S{i}]\nSection: {res['metadata']['section_number']} - {res['metadata']['section_title']}\n"
                f"Pages: {pages}\nText:\n{res['text']}"
            )
        
        context = "\n\n".join(context_parts)
        system_prompt = "You are a careful medical assistant. Answer ONLY using the provided context. Every claim must have a citation like [S1]."
        user_prompt = f"Question: {query}\n\nContext:\n{context}\n\nAnswer:"
        
        return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

    def generate(self, messages):
        inputs = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=Config.MAX_NEW_TOKENS, do_sample=False, pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True).strip()
