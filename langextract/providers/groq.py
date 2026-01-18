# Copyright 2025 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Groq provider for LangExtract."""

from __future__ import annotations

import dataclasses
import os
from typing import Any

from langextract.core import data
from langextract.providers import patterns
from langextract.providers import router
from langextract.providers.openai import OpenAILanguageModel


@router.register(
    *patterns.GROQ_PATTERNS,
    priority=patterns.GROQ_PRIORITY,
)
@dataclasses.dataclass(init=False)
class GroqLanguageModel(OpenAILanguageModel):
  """Language model inference using Groq's OpenAI-compatible API."""

  def __init__(
      self,
      model_id: str = 'groq/llama3-8b-8192',
      api_key: str | None = None,
      base_url: str | None = None,
      organization: str | None = None,
      format_type: data.FormatType = data.FormatType.JSON,
      temperature: float | None = None,
      max_workers: int = 10,
      **kwargs,
  ) -> None:
    """Initialize the Groq language model.

    Args:
      model_id: The Groq model ID (e.g., 'groq/llama3-8b-8192').
                The 'groq/' prefix is optional but recommended for auto-routing.
                It will be stripped before calling the API.
      api_key: API key for Groq service. Defaults to GROQ_API_KEY env var.
      base_url: Base URL for Groq service. Defaults to https://api.groq.com/openai/v1.
      organization: Optional organization ID.
      format_type: Output format (JSON or YAML).
      temperature: Sampling temperature.
      max_workers: Maximum number of parallel API calls.
      **kwargs: Extra parameters passed to the underlying OpenAI client.
    """
    # Strip groq/ prefix if present
    if model_id.startswith('groq/'):
      model_id = model_id[5:]

    # Default to Groq's API URL
    if base_url is None:
      base_url = "https://api.groq.com/openai/v1"

    # Default to GROQ_API_KEY environment variable
    if api_key is None:
      api_key = os.environ.get("GROQ_API_KEY")

    super().__init__(
        model_id=model_id,
        api_key=api_key,
        base_url=base_url,
        organization=organization,
        format_type=format_type,
        temperature=temperature,
        max_workers=max_workers,
        **kwargs,
    )
