"""Synthetic prompts for the load runs.

Every prompt is an instruction followed by a made-up document built from a fixed word list with a
seeded random generator, so every run and every arm sends the same prompts. The documents are
synthetic: they describe nothing real.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

DEFAULT_SEED = 20261024
DEFAULT_COUNT = 256
MIN_WORDS = 100
MAX_WORDS = 400

NOUNS = [
    "harbor", "lantern", "ledger", "orchard", "bridge", "workshop", "station", "archive", "garden",
    "market", "tower", "canal", "village", "library", "engine", "compass", "bakery", "quarry",
    "meadow", "lighthouse", "foundry", "pier", "granary", "observatory", "tunnel", "fountain",
    "courtyard", "mill", "clinic", "depot", "greenhouse", "ferry", "kiln", "cellar", "chapel",
    "cart", "lamp", "bench", "gate", "well", "farmer", "clerk", "sailor", "painter", "teacher",
    "baker", "surveyor", "mason", "weaver", "courier", "guide", "keeper", "printer", "carpenter",
    "botanist", "navigator", "tailor", "potter", "miller",
]
ADJECTIVES = [
    "quiet", "narrow", "crowded", "ancient", "bright", "damp", "tidy", "windy", "distant", "small",
    "busy", "faded", "heavy", "gentle", "restless", "careful", "plain", "weathered", "sunny",
    "hidden", "steady", "cold", "warm", "new", "crooked", "silent", "wide", "patient", "curious",
    "stubborn", "modest", "lively", "dusty", "bare", "green", "grey", "golden", "worn", "swift",
    "calm",
]
VERBS = [
    "repaired", "carried", "counted", "painted", "measured", "watched", "cleaned", "mapped",
    "wrapped", "sorted", "opened", "closed", "followed", "lifted", "moved", "recorded", "visited",
    "described", "polished", "tested", "folded", "delivered", "inspected", "arranged", "noticed",
    "borrowed", "returned", "stacked", "mended", "guarded",
]
PLACES = [
    "old harbor", "north gate", "river bend", "town square", "east field", "stone bridge",
    "clock tower", "salt flats", "mountain road", "back lane", "market hall", "low hill",
    "west dock", "pine wood", "chalk cliff", "reed marsh", "main street", "quiet cove",
    "iron works", "orchard wall",
]
NUMBERS = ["two", "three", "four", "five", "six", "seven", "eight", "nine", "twelve", "twenty"]

SENTENCES = [
    "The {adj} {noun} {verb} the {adj2} {noun2} near the {place}.",
    "Near the {place}, a {noun} {verb} {number} {adj} lamps.",
    "A {adj} {noun} and a {adj2} {noun2} {verb} the {noun3} by the {place}.",
    "Each morning the {noun} {verb} the {adj} {noun2} at the {place}.",
    "{number} {adj} visitors {verb} the {noun} before the {noun2} reached the {place}.",
    "Later, the {adj} {noun} {verb} {number} crates beside the {place}.",
    "It was the {adj} {noun} who {verb} the {noun2}, not the {noun3}.",
    "Beyond the {place}, the {adj2} {noun} {verb} everything the {noun2} had left.",
]

INSTRUCTIONS = [
    "Summarize the following notes in three sentences.",
    "List the main events in the following notes, one per line.",
    "Rewrite the following notes as a short news report.",
    "Explain the following notes to a ten-year-old.",
    "Write a short poem inspired by the following notes.",
    "Describe what a reader should remember from the following notes.",
    "Continue the following notes with what could happen next.",
    "Write three questions a reader could ask about the following notes.",
]


@dataclass(frozen=True)
class Prompt:
    id: int
    text: str
    n_words: int


def _sentence(rng: random.Random) -> str:
    template = rng.choice(SENTENCES)
    return template.format(
        adj=rng.choice(ADJECTIVES), adj2=rng.choice(ADJECTIVES),
        noun=rng.choice(NOUNS), noun2=rng.choice(NOUNS), noun3=rng.choice(NOUNS),
        verb=rng.choice(VERBS), place=rng.choice(PLACES), number=rng.choice(NUMBERS),
    ).capitalize()


def build_prompts(count: int = DEFAULT_COUNT, seed: int = DEFAULT_SEED,
                  min_words: int = MIN_WORDS, max_words: int = MAX_WORDS) -> list[Prompt]:
    """`count` deterministic prompts whose documents run from min_words to about max_words words."""
    if count <= 0:
        raise ValueError("count must be positive")
    if not 0 < min_words <= max_words:
        raise ValueError("need 0 < min_words <= max_words")
    rng = random.Random(seed)
    prompts = []
    for i in range(count):
        target = rng.randint(min_words, max_words)
        sentences: list[str] = []
        words = 0
        while words < target:
            s = _sentence(rng)
            sentences.append(s)
            words += len(s.split())
        text = f"{rng.choice(INSTRUCTIONS)}\n\n{' '.join(sentences)}"
        prompts.append(Prompt(id=i, text=text, n_words=len(text.split())))
    return prompts


def chat_body(text: str, *, model: str, max_tokens: int, fixed_length: bool,
              temperature: float = 0.0, stream: bool = True) -> dict:
    """Body of an OpenAI-style chat completion request.

    `fixed_length` adds vLLM's `ignore_eos`, so a request produces exactly `max_tokens` tokens.
    """
    body = {
        "model": model,
        "messages": [{"role": "user", "content": text}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }
    if stream:
        body["stream_options"] = {"include_usage": True}
    if fixed_length:
        body["ignore_eos"] = True
    return body
