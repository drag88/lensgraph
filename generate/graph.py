"""LangGraph wiring for the generation loop.

The graph has six nodes (Plan → Retrieve → Rerank → Verify →
[loop back to Retrieve OR continue] → Generate → Cite → END) and one
conditional edge out of Verify.

Iteration cap = 2 — a single retry covers the common
"model-needs-to-re-search" case; two covers the rare cascade. Beyond
that the trace yields an abstention rather than spinning.

Every node is wrapped in a span context manager (``generate.trace.span``)
so the trace_spans table receives one row per execution, including loop
re-entries. Spans batch-flush in ``api.answer`` after the graph returns.

The graph factory takes a ``conn`` and a span ``buffer`` so the nodes
that need them (retrieve, rerank, verify) can stay (AgentState) →
AgentState at the LangGraph layer.
"""

from __future__ import annotations

from collections.abc import Callable

from langgraph.graph import END, StateGraph

from generate.nodes import cite as cite_mod
from generate.nodes import generate as generate_mod
from generate.nodes import plan as plan_mod
from generate.nodes import rerank as rerank_mod
from generate.nodes import retrieve as retrieve_mod
from generate.nodes import verify as verify_mod
from generate.state import AgentState
from generate.trace import serialize_state_snapshot, span


def _wrap(
    fn: Callable[[AgentState], AgentState],
    *,
    node_name: str,
    buffer: list[dict],
) -> Callable[[AgentState], AgentState]:
    """Return a (state) -> state function that emits a span and runs
    ``fn``. ``iteration`` is read off the state at call time so the
    Retrieve loop's second pass records ``iteration=1``."""

    def wrapped(state: AgentState) -> AgentState:
        trace_id = state.get("trace_id", "")
        iteration = int(state.get("iteration", 0))
        input_snap = serialize_state_snapshot(state)
        with span(
            buffer,
            trace_id=trace_id,
            node_name=node_name,
            iteration=iteration,
            input_snapshot=input_snap,
        ) as carrier:
            new_state = fn(state)
            carrier["output_snapshot"] = serialize_state_snapshot(new_state)
            return new_state

    return wrapped


def build_graph(*, conn, span_buffer: list[dict]):
    """Compile the LangGraph for one ``answer()`` invocation.

    ``conn`` is baked into the nodes that need it via a closure;
    ``span_buffer`` collects span dicts across the run for the
    post-graph flush in ``api.answer``.
    """

    # Nodes that need conn: retrieve, rerank, verify.
    def _retrieve(state: AgentState) -> AgentState:
        return retrieve_mod.retrieve(state, conn=conn)

    def _rerank(state: AgentState) -> AgentState:
        return rerank_mod.rerank(state, conn=conn)

    def _verify(state: AgentState) -> AgentState:
        return verify_mod.verify(state, conn=conn)

    g = StateGraph(AgentState)
    g.add_node("plan", _wrap(plan_mod.plan, node_name="plan", buffer=span_buffer))
    g.add_node("retrieve", _wrap(_retrieve, node_name="retrieve", buffer=span_buffer))
    g.add_node("rerank", _wrap(_rerank, node_name="rerank", buffer=span_buffer))
    g.add_node("verify", _wrap(_verify, node_name="verify", buffer=span_buffer))
    g.add_node(
        "generate",
        _wrap(generate_mod.generate, node_name="generate", buffer=span_buffer),
    )
    g.add_node("cite", _wrap(cite_mod.cite, node_name="cite", buffer=span_buffer))

    g.set_entry_point("plan")
    g.add_edge("plan", "retrieve")
    g.add_edge("retrieve", "rerank")
    g.add_edge("rerank", "verify")

    def verify_router(state: AgentState) -> str:
        """Loop back to Retrieve when the judge is unsatisfied AND the judge
        proposed a refinement (``refined_query`` not None). The iteration
        cap lives in ``verify``: it sets ``refined_query=None`` once
        ``state['iteration']`` has reached the limit, which terminates the
        loop here. Keeping a single source of truth for the cap avoids
        off-by-one drift between the two checks.

        With ``iteration`` counting completed Verify→Retrieve LOOPS and
        the cap at ``current_iter < 2`` in ``verify``, the retrieve span
        sequence is ``[0, 1, 2]`` (initial + 2 loops); the 3rd verify
        fires the cap, the router exits to Generate. Matches design §5
        "two covers the rare cascade" and test (c)'s
        ``max(retrieve_iters) == 2``."""
        if (
            state.get("verify_confidence", 1.0) < state["verify_confidence_threshold"]
            and state.get("refined_query") is not None
        ):
            return "retrieve"
        return "generate"

    g.add_conditional_edges(
        "verify",
        verify_router,
        {"retrieve": "retrieve", "generate": "generate"},
    )
    g.add_edge("generate", "cite")
    g.add_edge("cite", END)
    return g.compile()
