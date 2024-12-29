import logging
from r2r import R2RException
from typing import AsyncGenerator

class StreamHandler:
    
    def __init__(self):
        self._inside_completion = False
        self._logger = logging.getLogger(__name__)

    async def process_stream(self, stream: AsyncGenerator[str, None]) -> AsyncGenerator[str, None]:
        """
        Process the R2R async stream token by token, yielding only data within <completion> tags.

        Args:
            stream (AsyncGenerator[str, None]): The async token generator from the LLM method.

        Yields:
            str: Completion text tokens.
        """
        async for chunk in stream:
            try:
                # Check if we're entering completion section
                if '<completion>' in chunk:
                    self._inside_completion = True
                    # Extract and yield content after <completion>
                    after_tag = chunk.split('<completion>', 1)[1]
                    
                    # If there's also a closing tag in the same chunk
                    if '</completion>' in after_tag:
                        before_close, _ = after_tag.split('</completion>', 1)
                        if before_close:
                            yield before_close
                        self._inside_completion = False
                    else:
                        # No closing tag yet, yield what we have
                        if after_tag:
                            yield after_tag
                    continue

                # If we're inside completion and looking for a closing tag
                if self._inside_completion:
                    if '</completion>' in chunk:
                        before_close, _ = chunk.split('</completion>', 1)
                        if before_close:
                            yield before_close
                        self._inside_completion = False
                    else:
                        if chunk:
                            yield chunk
                            
            except R2RException as r2re:
                self._logger.error(f"Error processing chunk: {r2re}")
                continue
            
            except Exception as e:
                self._logger.error(f"Error processing chunk: {e}")
                continue