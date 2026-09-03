from typing import Literal


def decide_route(input_text: str) -> Literal["refund", "support", "general"]:
    """Simple conditional routing logic."""
    text = input_text.lower()

    if "refund" in text or "return" in text:
        return "refund"
    if "billing" in text or "charge" in text or "payment" in text:
        return "support"
    return "general"


def refund_handler() -> str:
    return "Process refund request and check payment details."


def support_handler() -> str:
    return "Connect to support team and review billing issue."


def general_handler() -> str:
    return "Provide general assistance and answer the user query."


def workflow(user_message: str) -> str:
    """Run a conditional workflow based on user intent."""
    route = decide_route(user_message)

    if route == "refund":
        return refund_handler()
    if route == "support":
        return support_handler()
    return general_handler()


if __name__ == "__main__":
    samples = [
        "I want a refund for my order",
        "My charge is incorrect",
        "Tell me about your product",
    ]

    for sample in samples:
        print(f"User: {sample}")
        print(f"Route: {decide_route(sample)}")
        print(f"Result: {workflow(sample)}")
        print("-" * 40)
