import pytest

from servebench import workload


def test_prompts_are_deterministic_for_a_seed():
    a = workload.build_prompts(32, seed=1)
    b = workload.build_prompts(32, seed=1)
    c = workload.build_prompts(32, seed=2)
    assert [p.text for p in a] == [p.text for p in b]
    assert [p.text for p in a] != [p.text for p in c]


def test_default_set_is_256_distinct_prompts_in_the_word_range():
    prompts = workload.build_prompts()
    assert len(prompts) == workload.DEFAULT_COUNT == 256
    assert len({p.text for p in prompts}) == 256
    assert [p.id for p in prompts] == list(range(256))
    # the target is a lower bound on the document; the last sentence can overshoot a little
    assert min(p.n_words for p in prompts) >= workload.MIN_WORDS
    assert max(p.n_words for p in prompts) <= workload.MAX_WORDS + 40


def test_prompt_starts_with_an_instruction_then_the_document():
    p = workload.build_prompts(4)[0]
    instruction, document = p.text.split("\n\n", 1)
    assert instruction in workload.INSTRUCTIONS
    assert document.endswith(".")


def test_build_prompts_validates_arguments():
    with pytest.raises(ValueError):
        workload.build_prompts(0)
    with pytest.raises(ValueError):
        workload.build_prompts(1, min_words=50, max_words=10)


def test_chat_body_fixed_length_sets_ignore_eos_and_usage_option():
    body = workload.chat_body("hi", model="bench", max_tokens=128, fixed_length=True)
    assert body["ignore_eos"] is True
    assert body["max_tokens"] == 128
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}
    assert body["messages"] == [{"role": "user", "content": "hi"}]
    assert body["temperature"] == 0.0


def test_chat_body_free_length_has_no_ignore_eos():
    body = workload.chat_body("hi", model="bench", max_tokens=8, fixed_length=False)
    assert "ignore_eos" not in body
