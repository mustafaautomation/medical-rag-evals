"""Import shim for ragas 0.4.3 on current langchain-community.

ragas 0.4.3 imports ``ChatVertexAI`` from
``langchain_community.chat_models.vertexai`` at module import time. That
module was removed from langchain-community 0.4. The class now lives in the
optional ``langchain-google-vertexai`` package. This clinic demo does not call
Vertex AI. The placeholder is only here so the rest of ragas can import.
"""

import sys
import types
import warnings


def install_ragas_import_shim() -> None:
    module_name = "langchain_community.chat_models.vertexai"
    if module_name in sys.modules:
        return
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            __import__(module_name)
    except ModuleNotFoundError:
        module = types.ModuleType(module_name)
        module.ChatVertexAI = type("ChatVertexAI", (), {})  # type: ignore[attr-defined]
        sys.modules[module_name] = module


install_ragas_import_shim()
