"""ConversationManager — owns the LLM message list for one talkback session.

Multi-turn within a session; discarded when session ends. No cross-session persistence.
"""


class ConversationManager:
    """Tracks user/assistant message history with a fixed system prompt."""

    def __init__(self, system_prompt: str):
        self._system_prompt = system_prompt
        self._messages: list[dict[str, str]] = []
        self._turn_count = 0

    @property
    def turn_count(self) -> int:
        return self._turn_count

    def add_user_turn(self, text: str) -> None:
        self._messages.append({"role": "user", "content": text})

    def add_assistant_turn(self, text: str) -> None:
        self._messages.append({"role": "assistant", "content": text})
        self._turn_count += 1

    def get_messages(self) -> list[dict[str, str]]:
        return [{"role": "system", "content": self._system_prompt}] + self._messages

    def reset(self) -> None:
        self._messages.clear()
        self._turn_count = 0
