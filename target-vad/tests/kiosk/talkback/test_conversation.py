"""Tests for ConversationManager — multi-turn message list for one session."""

from modes.talkback.conversation import ConversationManager


class TestSystemPrompt:
    def test_system_prompt_is_first_message(self):
        cm = ConversationManager(system_prompt="You are helpful.")
        msgs = cm.get_messages()
        assert len(msgs) == 1
        assert msgs[0] == {"role": "system", "content": "You are helpful."}

    def test_system_prompt_persists_across_turns(self):
        cm = ConversationManager(system_prompt="Be concise.")
        cm.add_user_turn("hello")
        cm.add_assistant_turn("hi")
        msgs = cm.get_messages()
        assert msgs[0] == {"role": "system", "content": "Be concise."}


class TestTurnAlternation:
    def test_user_then_assistant(self):
        cm = ConversationManager(system_prompt="sys")
        cm.add_user_turn("what time is it")
        cm.add_assistant_turn("I don't have a clock.")
        msgs = cm.get_messages()
        assert msgs[1] == {"role": "user", "content": "what time is it"}
        assert msgs[2] == {"role": "assistant", "content": "I don't have a clock."}

    def test_multiple_exchanges(self):
        cm = ConversationManager(system_prompt="sys")
        cm.add_user_turn("a")
        cm.add_assistant_turn("b")
        cm.add_user_turn("c")
        cm.add_assistant_turn("d")
        msgs = cm.get_messages()
        assert len(msgs) == 5
        roles = [m["role"] for m in msgs]
        assert roles == ["system", "user", "assistant", "user", "assistant"]


class TestTurnCount:
    def test_turn_count_zero_initially(self):
        cm = ConversationManager(system_prompt="sys")
        assert cm.turn_count == 0

    def test_turn_count_after_exchange(self):
        cm = ConversationManager(system_prompt="sys")
        cm.add_user_turn("hi")
        cm.add_assistant_turn("hello")
        assert cm.turn_count == 1

    def test_turn_count_after_multiple(self):
        cm = ConversationManager(system_prompt="sys")
        cm.add_user_turn("a")
        cm.add_assistant_turn("b")
        cm.add_user_turn("c")
        cm.add_assistant_turn("d")
        assert cm.turn_count == 2


class TestReset:
    def test_reset_clears_history(self):
        cm = ConversationManager(system_prompt="sys")
        cm.add_user_turn("hi")
        cm.add_assistant_turn("hello")
        cm.reset()
        msgs = cm.get_messages()
        assert len(msgs) == 1
        assert msgs[0]["role"] == "system"

    def test_reset_resets_turn_count(self):
        cm = ConversationManager(system_prompt="sys")
        cm.add_user_turn("hi")
        cm.add_assistant_turn("hello")
        cm.reset()
        assert cm.turn_count == 0


class TestPendingUserTurn:
    def test_get_messages_for_llm_includes_pending_user(self):
        cm = ConversationManager(system_prompt="sys")
        cm.add_user_turn("what's up")
        msgs = cm.get_messages()
        assert msgs[-1] == {"role": "user", "content": "what's up"}
