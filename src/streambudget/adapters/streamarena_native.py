"""Optional native-driver wrapper. Import only inside StreamArena's driver environment.

The pinned driver imports its base class as the top-level module ``agent``.
Keeping that dependency here leaves the standalone StreamBudget package independent.
"""

from agent import StreamingAgent

from .streamarena import StreamBudgetAgent


class NativeStreamBudgetAgent(StreamBudgetAgent, StreamingAgent):
    """Satisfies the upstream loader's isinstance check without vendoring its code."""
