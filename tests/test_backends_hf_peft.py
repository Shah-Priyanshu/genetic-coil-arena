"""Test HFPeftBackend on a tiny random Qwen3 (CPU, float32, no downloads).

Skipped where torch, transformers or peft are missing (e.g. the lean CI venv).
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from genetic_coil_arena.backends import (  # noqa: E402
    HFPeftBackend,
    KVPrefix,
    LatentBackend,
    kv_bytes_per_token,
)

CHARS = "abcdefghijklmnopqrstuvwxyz0123456789 .,:{}\"'\n-"


def make_tokenizer():
    from tokenizers import Regex, Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {"<pad>": 0, "<eos>": 1, "<unk>": 2}
    for ch in CHARS:
        vocab[ch] = len(vocab)
    tk = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Split(Regex("."), behavior="isolated")
    return PreTrainedTokenizerFast(
        tokenizer_object=tk, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>"
    )


@pytest.fixture(scope="module")
def backend():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    tok = make_tokenizer()
    cfg = Qwen3Config(
        vocab_size=len(tok),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=256,
        pad_token_id=0,
        eos_token_id=1,
        tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(cfg).float()
    be = HFPeftBackend(
        "tiny-qwen3", device="cpu", dtype="float32", model=model, tokenizer=tok
    )
    be.add_adapter("a", r=4, alpha=8, seed=1)
    be.add_adapter("b", r=4, alpha=8, seed=2)
    be.add_adapter("c", r=4, alpha=8, seed=3)
    # LoRA B starts at zero; give "a" and "b" real weights so they differ from the base
    for name, seed in (("a", 11), ("b", 12)):
        gen = torch.Generator().manual_seed(seed)
        for p in be.adapter_parameters(name):
            p.data.copy_(torch.randn(p.shape, generator=gen) * 0.3)
    return be


PROMPTS = ["abc", "hello world 12", "x"]


def logits_of(be, prompt, adapter):
    with torch.no_grad(), be.use_adapter(adapter):
        inputs = be.build_inputs([prompt])
        hidden = be.forward_hidden(inputs).last_hidden_state
        return be._lm().lm_head(hidden)


def test_satisfies_latent_backend_protocol(backend):
    assert isinstance(backend, LatentBackend)
    assert backend.num_layers == 3 and backend.hidden_size == 32
    assert backend.adapters == ["a", "b", "c"]


def test_fresh_adapter_equals_base_and_trained_one_does_not(backend):
    base = logits_of(backend, "hello", None)
    assert torch.allclose(logits_of(backend, "hello", "c"), base, atol=1e-6)
    assert not torch.allclose(logits_of(backend, "hello", "a"), base, atol=1e-3)
    assert not torch.allclose(
        logits_of(backend, "hello", "a"), logits_of(backend, "hello", "b"), atol=1e-3
    )


def test_use_adapter_restores_previous_selection(backend):
    backend.set_adapter("a")
    with backend.use_adapter("b"):
        assert backend.adapter_id == "b"
        with backend.use_adapter(None):
            assert backend.adapter_id is None
        assert backend.adapter_id == "b"
    assert backend.adapter_id == "a"
    with pytest.raises(KeyError):
        backend.set_adapter("nope")
    with pytest.raises(ValueError):
        backend.add_adapter("a.b")


def test_generate_left_padding_matches_single_prompts(backend):
    batched = backend.generate(PROMPTS, "a", max_new_tokens=6)
    for prompt, result in zip(PROMPTS, batched):
        (alone,) = backend.generate([prompt], "a", max_new_tokens=6)
        assert result.token_ids == alone.token_ids
        assert result.text == alone.text
        assert result.n_new_tokens == len(result.token_ids)
    assert batched[1].n_prompt_tokens == len("hello world 12")


def test_generate_stops_at_eos_and_pads_after(backend):
    for r in backend.generate(PROMPTS, "a", max_new_tokens=5, stop_ids=[]):
        assert r.finish_reason == "length" and r.n_new_tokens == 5
    # Declare the token a row would produce first as the stop token: generation ends there
    first = backend.generate(["abc"], "a", max_new_tokens=3, stop_ids=[])[0].token_ids[
        0
    ]
    (stopped,) = backend.generate(["abc"], "a", max_new_tokens=3, stop_ids=[first])
    assert stopped.finish_reason == "stop" and stopped.n_new_tokens == 1
    assert stopped.text == ""  # the stop token is not part of the text


def test_hidden_states_layers_and_padding(backend):
    hs = backend.hidden_states(PROMPTS, 1, "a")
    assert hs.states.shape == (3, len("hello world 12"), 32)
    assert hs.mask.sum(-1).tolist() == [3, 14, 1]
    assert hs.layer == 1
    # Left padded: the last position of a short prompt equals that prompt run alone
    alone = backend.hidden_states(["abc"], 1, "a")
    assert torch.allclose(hs.states[0, -1], alone.states[0, -1], atol=1e-5)
    ids = torch.tensor(
        [backend.tokenizer("abc", add_special_tokens=False)["input_ids"]]
    )
    embeddings = backend.hidden_states(["abc"], 0, "a").states
    assert torch.allclose(embeddings, backend.embed(ids), atol=1e-6)
    assert backend.hidden_states(["abc"], -1, "a").layer == backend.num_layers
    with pytest.raises(ValueError):
        backend.hidden_states(["abc"], 99)


def test_selective_loss_matches_full_vocabulary_loss(backend):
    prompts, targets = ["abc:", "hello:"], ["12", "world"]
    with torch.no_grad(), backend.use_adapter("a"):
        inputs = backend.build_inputs(prompts, targets)
        loss = backend.training_loss(inputs)
        logits = backend._lm()(
            inputs_embeds=inputs.inputs_embeds,
            attention_mask=inputs.attention_mask,
            position_ids=inputs.position_ids,
        ).logits.float()
    full = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]),
        inputs.labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )
    assert torch.allclose(loss, full, atol=1e-5)
    # labels: only the target and its stop token are supervised
    assert (inputs.labels != -100).sum().item() == len("12") + len("world") + 2


def test_chunked_loss_matches_unchunked_value_and_gradient(backend):
    prompts, targets = ["abc:", "hello:", "x:"], ["12345", "world", "9"]
    backend.set_adapter("a", trainable=True)
    try:
        grads = {}
        for chunk in (None, 4):
            for p in backend.adapter_parameters("a"):
                p.grad = None
            inputs = backend.build_inputs(prompts, targets)
            hidden = backend.forward_hidden(inputs).last_hidden_state
            loss = backend.lm_loss(hidden, inputs.labels, chunk_size=chunk)
            loss.backward()
            grads[chunk] = (
                loss.detach(),
                torch.cat([p.grad.flatten() for p in backend.adapter_parameters("a")]),
            )
    finally:
        backend.set_adapter("a", trainable=False)
        for p in backend.adapter_parameters("a"):
            p.grad = None
    assert torch.allclose(grads[None][0], grads[4][0], atol=1e-6)
    assert torch.allclose(grads[None][1], grads[4][1], atol=1e-5)
    assert grads[4][1].abs().sum() > 0


def test_soft_tokens_are_prepended_and_used(backend):
    zeros = torch.zeros(2, 3, 32)
    noise = torch.randn(2, 3, 32)
    inputs = backend.build_inputs(["abc", "de"], soft=noise)
    assert inputs.inputs_embeds.shape[1] == 3 + 3
    assert inputs.attention_mask[:, :3].all()
    assert inputs.n_prompt_tokens.tolist() == [6, 5]
    assert (inputs.labels[:, :3] == -100).all()
    a = backend.generate(["abc", "de"], "a", max_new_tokens=4, soft=zeros)
    b = backend.generate(["abc", "de"], "a", max_new_tokens=4, soft=noise * 5)
    assert [r.token_ids for r in a] != [r.token_ids for r in b]
    with pytest.raises(ValueError):
        backend.build_inputs(["abc"], soft=noise)


def test_kv_prefix_equals_running_the_concatenation(backend):
    senders, receivers = ["abc", "hello world"], [" -> ok", " -> ok"]
    sender_adapter, receiver_adapter = "a", "a"  # same weights: prefix must be exact
    prefix = backend.prefill(senders, sender_adapter)
    assert isinstance(prefix, KVPrefix)
    assert prefix.lengths.tolist() == [3, 11]
    with torch.no_grad(), backend.use_adapter(receiver_adapter):
        inputs = backend.build_inputs(receivers, prefix=prefix, padding_side="right")
        got = backend._lm().lm_head(backend.forward_hidden(inputs).last_hidden_state)
        for i, (s, r) in enumerate(zip(senders, receivers)):
            joint = backend.build_inputs([s + r])
            ref = backend._lm().lm_head(backend.forward_hidden(joint).last_hidden_state)
            assert torch.allclose(got[i], ref[0, len(s) :], atol=1e-4)


def test_prefix_is_cloned_and_generation_runs_on_it(backend):
    prefix = backend.prefill(["hello world"], "a")
    before = prefix.cache.get_seq_length()
    out = backend.generate([" ->"], "b", max_new_tokens=4, prefix=prefix)
    assert out[0].n_new_tokens == 4 and prefix.cache.get_seq_length() == before
    again = backend.generate([" ->"], "b", max_new_tokens=4, prefix=prefix)
    assert again[0].token_ids == out[0].token_ids  # same prefix, same continuation


def test_map_kv_identity_and_zero(backend):
    prefix = backend.prefill(["hello world"], "a")
    same = backend.map_kv(prefix, lambda layer, kind, t: t.clone())
    zero = backend.map_kv(prefix, lambda layer, kind, t: torch.zeros_like(t))
    with torch.no_grad(), backend.use_adapter("b"):

        def run(p):
            inputs = backend.build_inputs([" ->"], prefix=p, padding_side="right")
            return backend.forward_hidden(inputs).last_hidden_state

        base = run(prefix)
        assert torch.allclose(base, run(same), atol=1e-6)
        assert not torch.allclose(base, run(zero), atol=1e-4)
    assert (
        backend.kv_nbytes(prefix.cache) == 11 * 3 * 2 * 2 * 8 * 4
    )  # tokens*layers*kv*heads*dim*f32
    assert backend.kv_bytes_per_token == kv_bytes_per_token(3, 2, 8, 4)


def test_fit_trains_only_its_adapter_even_with_a_frozen_sender_in_the_loop(backend):
    backend.add_adapter("learner", r=4, alpha=8, seed=5)
    backend.add_adapter("sender", r=4, alpha=8, seed=6)
    for p in backend.adapter_parameters("sender"):
        p.data.normal_(0, 0.2)
    snapshot = {n: backend.adapter_state(n) for n in ("a", "sender", "learner")}
    base_before = backend.model.model.embed_tokens.weight.clone()

    def step_fn(step):
        # a frozen sender runs between steps: it must not freeze the learner
        backend.hidden_states(["abc", "de"], 2, "sender")
        inputs = backend.build_inputs(["ab:", "cd:"], ["12", "34"])
        return backend.training_loss(inputs)

    stats = backend.fit("learner", step_fn, steps=40, lr=3e-2, warmup=2, log_every=10)
    first = sum(stats.losses[:3]) / 3
    assert stats.steps == 40 and len(stats.losses) == 40
    assert stats.mean_last_loss < first - 0.2
    assert backend.adapter_id == "learner"
    moved = backend.adapter_state("learner")
    assert any(not torch.equal(moved[k], v) for k, v in snapshot["learner"].items())
    for name in ("a", "sender"):
        now = backend.adapter_state(name)
        assert all(torch.equal(now[k], v) for k, v in snapshot[name].items())
    assert torch.equal(backend.model.model.embed_tokens.weight, base_before)
    assert not any(p.requires_grad for p in backend.adapter_parameters("learner"))
    for name in ("learner", "sender"):
        backend.delete_adapter(name)
    assert backend.adapters == ["a", "b", "c"]


def test_fit_trains_extra_parameters_and_rejects_nan(backend):
    backend.add_adapter("rx", r=4, alpha=8, seed=7)
    proj = torch.nn.Linear(32, 32)
    before = proj.weight.clone()

    def step_fn(step):
        soft = proj(torch.randn(2, 2, 32))
        inputs = backend.build_inputs(["ab:", "cd:"], ["12", "34"], soft=soft)
        return backend.training_loss(inputs)

    backend.fit(
        "rx", step_fn, steps=3, lr=1e-2, extra_parameters=list(proj.parameters())
    )
    assert not torch.equal(proj.weight, before)
    with pytest.raises(RuntimeError, match="loss is"):
        backend.fit(
            "rx", lambda s: torch.tensor(float("nan"), requires_grad=True), steps=1
        )
    backend.delete_adapter("rx")


def test_fit_can_train_a_sender_adapter_end_to_end(backend):
    backend.add_adapter("rx2", r=4, alpha=8, seed=8)
    backend.add_adapter("tx2", r=4, alpha=8, seed=9)
    proj = torch.nn.Linear(32, 32)
    frozen_before = backend.adapter_state("a")
    tx_before = backend.adapter_state("tx2")

    def step_fn(step):
        states = backend.hidden_states(["abc", "de"], 2, "tx2", grad=True)
        assert states.states.requires_grad  # attached to the sender's adapter
        soft = proj(states.states[:, -2:])
        inputs = backend.build_inputs(["ab:", "cd:"], ["12", "34"], soft=soft)
        return backend.training_loss(inputs)

    stats = backend.fit(
        "rx2",
        step_fn,
        steps=5,
        lr=3e-2,
        extra_parameters=list(proj.parameters()),
        extra_adapters=["tx2"],
    )
    moved = backend.adapter_state("tx2")
    assert any(not torch.equal(moved[k], v) for k, v in tx_before.items())
    assert all(
        torch.equal(backend.adapter_state("a")[k], v) for k, v in frozen_before.items()
    )
    assert stats.n_trainable == sum(
        p.numel()
        for p in [
            *backend.adapter_parameters("rx2"),
            *backend.adapter_parameters("tx2"),
            *proj.parameters(),
        ]
    )
    # without grad the states are detached, and training afterwards leaves no stale flags
    assert not backend.hidden_states(["abc"], 2, "tx2").states.requires_grad
    assert not any(p.requires_grad for p in backend.adapter_parameters("tx2"))
    for name in ("rx2", "tx2"):
        backend.delete_adapter(name)


def test_adapter_save_load_round_trip(backend, tmp_path):
    path = tmp_path / "a.pt"
    backend.save_adapter("a", path)
    backend.load_adapter("a_copy", path)
    assert torch.allclose(
        logits_of(backend, "hello", "a_copy"),
        logits_of(backend, "hello", "a"),
        atol=1e-6,
    )
    backend.delete_adapter("a_copy")


def test_memory_helpers_are_safe_on_cpu(backend):
    assert backend.peak_memory_mib() == 0.0
    backend.reset_peak_memory()
    backend.synchronize()


def test_next_token_logprobs_are_true_log_probabilities(backend):
    prompts = ["abc", "hello world"]
    ids = [backend.tokenizer.encode(c, add_special_tokens=False)[0] for c in "abc"]
    lp = backend.next_token_logprobs(prompts, ids)
    assert len(lp) == 2 and all(len(row) == 3 for row in lp)
    assert all(x < 0 for row in lp for x in row)
    # the whole vocabulary sums to one, so the listed tokens sum to at most one
    assert all(sum(torch.tensor(row).exp().tolist()) <= 1 + 1e-5 for row in lp)
    # batch padding does not change a row's answer
    alone = backend.next_token_logprobs(["abc"], ids)[0]
    assert alone == pytest.approx(lp[0], abs=1e-4)
    # it agrees with the greedy choice: the argmax over the vocabulary is generated first
    full = backend.next_token_logprobs(["abc"], list(range(len(backend.tokenizer))))[0]
    first = backend.generate(["abc"], max_new_tokens=1)[0].token_ids[0]
    assert int(torch.tensor(full).argmax()) == first


def test_next_token_logprobs_chunked_prefill_matches_one_pass(backend):
    prompts = ["abc def ghi jkl mno", "hello world, this is longer text"]
    ids = list(range(3, 12))
    whole = backend.next_token_logprobs(prompts, ids)
    for chunk in (4, 7):
        chunked = backend.next_token_logprobs(prompts, ids, chunk_size=chunk)
        for a, b in zip(whole, chunked):
            assert a == pytest.approx(b, abs=1e-3)
