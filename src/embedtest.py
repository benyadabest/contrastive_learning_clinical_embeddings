"""
Tests if basic Basseten API call works 
"""

from __future__ import annotations

from baseten_performance_client import PerformanceClient, RequestProcessingPreference

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import os
from dotenv import load_dotenv

load_dotenv()

client = PerformanceClient(
    base_url="https://model-qrj9z513.api.baseten.co/environments/production/sync",
    api_key=os.getenv("BASETEN_API_KEY")
)

preference = RequestProcessingPreference(
    batch_size=16,
    max_concurrent_requests=256,
    max_chars_per_request=10000,
    hedge_delay=0.5,
    timeout_s=360,
    total_timeout_s=600
)

embed_response = client.embed(
    input=["text 1", "text 2"],
    model="library-model-embeddinggemma",
    preference=preference
)
print(f"Model used: {embed_response.model}")
print(f"Total tokens used: {embed_response.usage.total_tokens}")
print(f"Total time: {embed_response.total_time:.4f}s")
# Convert to numpy array (requires numpy)
numpy_array = embed_response.numpy()
print(f"Embeddings shape: {numpy_array.shape}")
