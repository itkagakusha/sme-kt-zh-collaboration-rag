import re
from enum import StrEnum

import pymupdf4llm

from conversational_toolkit.chunking.base import Chunk, Chunker


# TODO: Add other PDF to Markdown engines
# TODO: Handles images in PDFs


class MarkdownConverterEngine(StrEnum):
    PYMUPDF4LLM = "pymupdf4llm"


class PDFChunker(Chunker):
    def _pdf2markdown(
        self, file_path: str, engine: MarkdownConverterEngine = MarkdownConverterEngine.PYMUPDF4LLM
    ) -> str:
        if engine == MarkdownConverterEngine.PYMUPDF4LLM:
            return pymupdf4llm.to_markdown(file_path)  # type: ignore
        else:
            raise NotImplementedError("Specified engine is not supported.")

    def _normalize_newlines(self, text: str) -> str:
        paragraphs = text.split("\n\n")
        processed_paragraphs = [para.replace("\n", " ") for para in paragraphs]
        return "\n\n".join(processed_paragraphs)

    def make_chunks(
        self, file_path: str, engine: MarkdownConverterEngine = MarkdownConverterEngine.PYMUPDF4LLM
    ) -> list[Chunk]:
        markdown = self._pdf2markdown(file_path, engine)

        header_pattern = re.compile(r"^(#{1,6}\s.*)$", re.MULTILINE)
        matches = list(header_pattern.finditer(markdown))

        chunks: list[Chunk] = []
        current_chapters: list[str] = []

        if not matches:
            processed_text = self._normalize_newlines(markdown)
            chunk = Chunk(title="", content=processed_text, mime_type="text/markdown", metadata={"chapters": []})
            return [chunk]

        for i, match in enumerate(matches):
            header_line = match.group(1).strip()
            header_level = header_line.count("#", 0, header_line.find(" "))

            if len(current_chapters) < header_level:
                current_chapters.append(header_line)
            else:
                current_chapters = current_chapters[: header_level - 1] + [header_line]

            start_idx = match.start()
            if i < len(matches) - 1:
                end_idx = matches[i + 1].start()
            else:
                end_idx = len(markdown)
            chunk_text = markdown[start_idx:end_idx]

            processed_chunk_text = self._normalize_newlines(chunk_text)

            chunk = Chunk(
                title=header_line,
                content=processed_chunk_text,
                mime_type="text/markdown",
                metadata={"chapters": current_chapters.copy()},
            )
            chunks.append(chunk)

        return chunks
