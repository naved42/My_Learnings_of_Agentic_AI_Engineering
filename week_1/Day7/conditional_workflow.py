from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph


class LoanState(TypedDict):
    income: float
    decision: str


def routing(state: LoanState) -> Literal["approve", "reject"]:
    if state["income"] < 75000:
        return "reject"
    return "approve"


def loan_approved(state: LoanState) -> dict[str, str]:
    return {
        "decision": f"Loan approved successfully for income: {state['income']}"
    }


def loan_rejected(state: LoanState) -> dict[str, str]:
    return {
        "decision": f"Loan rejected for income: {state['income']}"
    }


graph = StateGraph(LoanState)
graph.add_node("approve", loan_approved)
graph.add_node("reject", loan_rejected)

graph.add_conditional_edges(START, routing)
graph.add_edge("approve", END)
graph.add_edge("reject", END)

workflow = graph.compile()

initial_state: LoanState = {
    "income": 25000.0,
    "decision": "",
}

final_state = workflow.invoke(initial_state)
print(final_state)
