import re
import logging
from typing import Generator

class R2RStreamHandler:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.inside_completion = False

    def process_stream(self, stream: Generator) -> Generator[str, None, None]:
        """
        Process the R2R stream token by token.
        
        Args:
            stream (Generator): Raw stream from R2R client's prompt_llm method
            
        Yields:
            str: Clean completion text tokens
        """
        for chunk in stream:
            try:
                # Check if we're entering completion section
                if '<completion>' in chunk:
                    self.inside_completion = True
                    # If there's content after the tag, yield it
                    content = chunk.split('<completion>')[1].strip()
                    if content:
                        yield content
                    continue

                # Check if we're exiting completion section, i.e. we have read all tokens
                if '</completion>' in chunk:
                    self.inside_completion = False
                    # If there's content before the tag, yield it
                    content = chunk.split('</completion>')[0].strip()
                    if content:
                        yield content
                    continue

                # If we're inside completion tags, yield the chunk
                if self.inside_completion and chunk.strip():
                    yield chunk

            except Exception as e:
                self.logger.error(f"Error processing chunk: {e}")
                continue