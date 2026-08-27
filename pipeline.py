import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .config import Config

class TextProcessor:
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def process_blocks(self, blocks_with_sections):
        chunks = []
        current_section = None
        current_text = []
        current_pages = []
        current_start_order = None
        current_metadata = None

        def save_section_chunks(text_list, pages_list, start_order, metadata, end_order):
            if not text_list: return
            full_text = "\n\n".join(text_list)
            section_chunks = self.splitter.split_text(full_text)
            for chunk_text in section_chunks:
                chunks.append({
                    "text": chunk_text,
                    "metadata": {**metadata, "pages": sorted(set(pages_list)), 
                                 "start_order": start_order, "end_order": end_order}
                })

        for block in blocks_with_sections:
            if block["section_number"] is None or block["type"] == "section_header": continue
            text = block["text"].strip()
            if not text: continue
            
            if current_section != block["section_number"]:
                if current_section is not None:
                    save_section_chunks(current_text, current_pages, current_start_order, current_metadata, block["order"] - 1)
                current_section = block["section_number"]
                current_text, current_pages = [], []
                current_start_order = block["order"]
                current_metadata = {
                    "section_id": block["section_id"], "section_number": block["section_number"],
                    "section_title": block["section_title"], "parent_section_id": block["parent_section_id"],
                    "depth": block["depth"]
                }
            current_text.append(text)
            if block["page"] is not None: current_pages.append(block["page"])

        if current_section is not None:
            save_section_chunks(current_text, current_pages, current_start_order, current_metadata, blocks_with_sections[-1]["order"])

        for i, chunk in enumerate(chunks): chunk["metadata"]["chunk_index"] = i
        return chunks
