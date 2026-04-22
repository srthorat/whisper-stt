"""LocalAgreement hypothesis buffer for streaming transcription.

Implements the LocalAgreement policy from whisper_streaming:
Words are committed (confirmed) only when they appear identically in two
consecutive transcription passes. Unstable trailing words stay as preview
until they stabilize.

Reference: https://github.com/ufal/whisper_streaming
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# Regex to strip trailing punctuation for comparison (preserves hyphens mid-word)
_TRAILING_PUNCT = re.compile(r'[.,!?;:…\'"»\])}]+$')


def _norm(text: str) -> str:
    """Normalize a word for LocalAgreement comparison.

    Strips whitespace and trailing punctuation so that e.g. ``"from"``
    and ``"from."`` are considered identical.
    """
    return _TRAILING_PUNCT.sub("", text.strip().lower())


@dataclass
class TimedWord:
    """A word with start/end timestamps (seconds) and probability."""

    start: float
    end: float
    text: str
    probability: float = 0.0


class HypothesisBuffer:
    """Manages confirmed and unconfirmed words via LocalAgreement.

    Three internal lists:
      committed_in_buffer — confirmed words still within the audio buffer
      buffer              — unconfirmed words from the *previous* transcription pass
      new                 — words from the *current* transcription pass

    flush() compares buffer and new word-by-word. The longest common prefix
    (identical words at the same position) is committed.  Everything after the
    first disagreement stays unconfirmed until the next pass.
    """

    def __init__(self) -> None:
        self.committed_in_buffer: list[TimedWord] = []
        self.buffer: list[TimedWord] = []
        self.new: list[TimedWord] = []
        self.last_committed_time: float = 0.0
        self.last_committed_word: str = ""

    # ------------------------------------------------------------------ #
    # Core algorithm
    # ------------------------------------------------------------------ #

    def insert(self, new_words: list[TimedWord], offset: float) -> None:
        """Register a new hypothesis from the latest transcription pass.

        Args:
            new_words: Word-level timestamped results from Whisper.
                       Timestamps are relative to the audio chunk start.
            offset:    buffer_time_offset to add to all timestamps so they
                       become segment-absolute.
        """
        # 1. Apply time offset
        shifted = [
            TimedWord(
                start=w.start + offset,
                end=w.end + offset,
                text=w.text,
                probability=w.probability,
            )
            for w in new_words
        ]

        # 2. Drop words that precede already-committed text (re-hallucinated)
        # Use wide tolerance (1.0 s) because Whisper timestamps shift between
        # passes on a growing buffer.  N-gram dedup (step 3) handles actual
        # duplicate prevention, so the time filter only needs to block words
        # that are clearly from the distant past.
        filtered = [w for w in shifted if w.start >= self.last_committed_time - 1.0]
        dropped = [w for w in shifted if w.start < self.last_committed_time - 1.0]
        if dropped:
            logger.debug("TIME-FILTER dropped: %s (last_committed=%.2f)",
                         [(w.text.strip(), f"{w.start:.2f}") for w in dropped],
                         self.last_committed_time)

        # 3. N-gram de-duplication against committed words (up to 8-grams)
        #    Prevents double-emitting when Whisper re-generates committed
        #    text from audio still in the buffer.
        if filtered and self.committed_in_buffer:
            if filtered[0].start <= self.last_committed_time + 2.0:
                for n in range(
                    min(8, len(self.committed_in_buffer), len(filtered)), 0, -1
                ):
                    tail = [_norm(w.text) for w in self.committed_in_buffer[-n:]]
                    head = [_norm(w.text) for w in filtered[:n]]
                    if tail == head:
                        filtered = filtered[n:]
                        break

        self.new = filtered
        logger.debug("INSERT new=[%s]", ", ".join(f'{w.text.strip()}({w.start:.2f})' for w in self.new))

    def flush(self) -> list[TimedWord]:
        """Apply LocalAgreement and return newly committed words.

        The longest common prefix (word text match) between *buffer*
        (previous pass) and *new* (current pass) is committed.
        Remaining *new* words become the next *buffer*.

        Enhancement: 1-word lookahead realignment.  If buffer and new
        disagree at some position, a bounded search (1 word ahead) in each
        list attempts to find alignment.  Buffer-side skipped words are
        committed (they were confirmed in a prior pass); new-side extra
        words are skipped without committing (need a second pass).
        """
        logger.debug("FLUSH buffer=[%s]", ", ".join(f'{w.text.strip()}' for w in self.buffer))
        committed: list[TimedWord] = []
        buf_idx = 0
        new_idx = 0

        while new_idx < len(self.new) and buf_idx < len(self.buffer):
            n_text = _norm(self.new[new_idx].text)
            b_text = _norm(self.buffer[buf_idx].text)
            if n_text == b_text:
                word = self.new[new_idx]
                committed.append(word)
                self.last_committed_time = word.end
                self.last_committed_word = word.text
                buf_idx += 1
                new_idx += 1
            else:
                logger.debug("DISAGREE buf[%d]=%r vs new[%d]=%r", buf_idx, b_text, new_idx, n_text)
                realigned = False

                # Case A: buffer has extra word not in new (prev-pass word dropped).
                # Commit the skipped word — it was confirmed in a prior pass.
                if (buf_idx + 1 < len(self.buffer)
                        and _norm(self.buffer[buf_idx + 1].text) == n_text):
                    skipped = self.buffer[buf_idx]
                    committed.append(skipped)
                    self.last_committed_time = skipped.end
                    self.last_committed_word = skipped.text
                    buf_idx += 1
                    logger.debug("SKIP-COMMIT buf word %r, realign", skipped.text.strip())
                    realigned = True

                # Case B: new has extra word not in buffer.
                # Skip without committing (needs a second pass to confirm).
                elif (new_idx + 1 < len(self.new)
                      and _norm(self.new[new_idx + 1].text) == b_text):
                    logger.debug("SKIP new word %r (unconfirmed), realign", self.new[new_idx].text.strip())
                    new_idx += 1
                    realigned = True

                if not realigned:
                    break

        # Left-over new words become pending for next pass
        self.buffer = self.new[new_idx:]
        self.new = []
        logger.debug("FLUSH committed=[%s] remaining_buffer=[%s]",
                     ", ".join(w.text.strip() for w in committed),
                     ", ".join(w.text.strip() for w in self.buffer))

        self.committed_in_buffer.extend(committed)
        return committed

    # ------------------------------------------------------------------ #
    # Buffer management helpers
    # ------------------------------------------------------------------ #

    def pop_committed(self, time: float) -> None:
        """Remove committed words whose end ≤ *time* (after audio trim)."""
        self.committed_in_buffer = [
            w for w in self.committed_in_buffer if w.end > time
        ]

    def complete(self) -> list[TimedWord]:
        """Return all unconfirmed words (end-of-speech flush)."""
        result = self.buffer
        self.buffer = []
        self.new = []
        return result

    def get_unconfirmed_text(self) -> str:
        """Current unconfirmed (preview) text."""
        return "".join(w.text for w in self.buffer).strip()

    def reset(self) -> None:
        """Reset for a new speech segment."""
        self.committed_in_buffer.clear()
        self.buffer.clear()
        self.new.clear()
        self.last_committed_time = 0.0
        self.last_committed_word = ""
