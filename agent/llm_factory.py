import os
from agent.g4f_wrapper import G4FChatModel

def get_llm(model_name: str = "deepseek-v3", temperature: float = 0):
    """
    Returns a configured G4F chat model using native g4f.client.
    """
    return G4FChatModel(
        model=model_name,
        temperature=temperature,
        web_search=True
    )
