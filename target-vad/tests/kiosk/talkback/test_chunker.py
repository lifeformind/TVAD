"""Tests for SentenceChunker — buffers LLM tokens, emits on sentence boundaries."""

from modes.talkback.chunker import SentenceChunker


class TestSentenceTerminators:
    def test_period_emits_chunk(self):
        c = SentenceChunker()
        chunks = []
        for token in ["Hello", " world", "."]:
            result = c.feed(token)
            if result:
                chunks.append(result)
        assert chunks == ["Hello world."]

    def test_question_mark_emits_chunk(self):
        c = SentenceChunker()
        chunks = []
        for token in ["How", " are", " you", "?"]:
            result = c.feed(token)
            if result:
                chunks.append(result)
        assert chunks == ["How are you?"]

    def test_exclamation_emits_chunk(self):
        c = SentenceChunker()
        chunks = []
        for token in ["Wow", "!"]:
            result = c.feed(token)
            if result:
                chunks.append(result)
        assert chunks == ["Wow!"]

    def test_multiple_sentences(self):
        c = SentenceChunker()
        chunks = []
        for token in ["First", ".", " Second", ".", " Third", "."]:
            result = c.feed(token)
            if result:
                chunks.append(result)
        assert chunks == ["First.", "Second.", "Third."]


class TestAbbreviations:
    def test_dr_does_not_emit(self):
        c = SentenceChunker()
        chunks = []
        for token in ["Dr", ".", " Smith", " is", " here", "."]:
            result = c.feed(token)
            if result:
                chunks.append(result)
        assert chunks == ["Dr. Smith is here."]

    def test_us_does_not_emit(self):
        c = SentenceChunker()
        chunks = []
        for token in ["The", " U", ".", "S", ".", " is", " big", "."]:
            result = c.feed(token)
            if result:
                chunks.append(result)
        assert chunks == ["The U.S. is big."]

    def test_mr_does_not_emit(self):
        c = SentenceChunker()
        chunks = []
        for token in ["Mr", ".", " Jones", " left", "."]:
            result = c.feed(token)
            if result:
                chunks.append(result)
        assert chunks == ["Mr. Jones left."]


class TestMaxChars:
    def test_max_chars_forces_emit(self):
        c = SentenceChunker(max_chunk_chars=20)
        chunks = []
        for token in ["This", " is", " a", " very", " long", " sentence", " without", " punctuation"]:
            result = c.feed(token)
            if result:
                chunks.append(result)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert len(chunk) <= 30


class TestFlush:
    def test_flush_emits_trailing_fragment(self):
        c = SentenceChunker()
        for token in ["No", " period", " here"]:
            c.feed(token)
        result = c.flush()
        assert result == "No period here"

    def test_flush_returns_none_when_empty(self):
        c = SentenceChunker()
        assert c.flush() is None

    def test_flush_after_sentence_returns_none(self):
        c = SentenceChunker()
        for token in ["Done", "."]:
            c.feed(token)
        assert c.flush() is None


class TestReset:
    def test_reset_clears_buffer(self):
        c = SentenceChunker()
        c.feed("partial")
        c.reset()
        assert c.flush() is None
