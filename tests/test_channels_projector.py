"""Test the latent projector and attention pool (needs torch)."""

import pytest

torch = pytest.importorskip("torch")


class TestProjector:
    D_IN, D_OUT, K = 32, 24, 3

    def projector(self, pooling="last", **kwargs):
        from genetic_coil_arena.channels import LatentProjector

        torch.manual_seed(0)
        return LatentProjector(
            self.D_IN, self.D_OUT, self.K, pooling=pooling, init_rms=0.05, **kwargs
        )

    def test_soft_tokens_start_at_embedding_scale_whatever_the_input_scale(self):
        proj = self.projector()
        for scale in (0.01, 1.0, 150.0):  # raw states are ~100x an embedding
            states = torch.randn(5, 10, self.D_IN) * scale
            soft = proj(states)
            assert soft.shape == (5, self.K, self.D_OUT)
            rms = soft.pow(2).mean(-1).sqrt()
            assert torch.allclose(rms, torch.full_like(rms, 0.05), rtol=0.02)

    def test_last_pooling_takes_the_last_k_positions(self):
        proj = self.projector()
        states = torch.randn(2, 7, self.D_IN)
        message = proj.encode(states)
        assert message.shape == (2, self.K, self.D_IN)
        assert torch.equal(message, proj.standardize(states[:, -self.K :]))
        assert torch.equal(proj.decode(message), proj(states))
        with pytest.raises(ValueError, match="at least k"):
            proj.encode(states[:, :2])

    def test_calibration_removes_massive_dimensions(self):
        torch.manual_seed(1)
        states = torch.randn(64, 10, self.D_IN)
        states[..., 3] = 400.0 + 20 * torch.randn(64, 10)  # a massive dimension
        states[..., 7] *= 50.0
        proj = self.projector()
        raw = proj.encode(states)
        proj.calibrate(states)
        message = proj.encode(states)
        assert message.shape == (64, self.K, self.D_IN)
        flat = message.reshape(-1, self.D_IN)
        assert torch.allclose(flat.mean(0), torch.zeros(self.D_IN), atol=0.15)
        assert torch.allclose(flat.std(0), torch.ones(self.D_IN), atol=0.15)
        assert raw.abs().max() <= proj.clip and message.abs().max() <= proj.clip
        # the calibration travels with the module
        other = self.projector()
        other.load_state_dict(proj.state_dict())
        assert torch.equal(other.encode(states), message)

    def test_calibration_with_attention_pooling_skips_padding_and_the_first_token(self):
        states = torch.randn(4, 12, self.D_IN)
        states[:, 0] = 1e4  # the attention sink
        mask = torch.ones(4, 12, dtype=torch.long)
        mask[:2, :3] = 0  # rows 0 and 1 are left-padded
        states[:2, :3] = 1e5  # junk under the mask
        states[:2, 3] = 1e4  # their first real token is the sink
        proj = self.projector("attn", n_heads=2, d_attn=16)
        proj.calibrate(states, mask)
        assert proj.norm_mean.abs().max() < 0.5
        assert (1 / proj.norm_scale).max() < 2.0  # std near 1, not near 1e4

    def test_attention_pooling_ignores_padding(self):
        proj = self.projector("attn", n_heads=2, d_attn=16)
        states = torch.randn(2, 9, self.D_IN)
        mask = torch.ones(2, 9, dtype=torch.long)
        mask[0, :4] = 0  # row 0 is left-padded by 4
        changed = states.clone()
        changed[0, :4] = torch.randn(4, self.D_IN) * 1000  # garbage under the mask
        a, b = proj.encode(states, mask), proj.encode(changed, mask)
        assert a.shape == (2, self.K, self.D_IN)
        assert torch.allclose(a, b, atol=1e-5)
        assert not torch.allclose(proj.encode(states), proj.encode(changed))
        # a different real position does change the message
        moved = states.clone()
        moved[0, 6] += 3.0
        assert not torch.allclose(proj.encode(moved, mask)[0], a[0], atol=1e-4)

    def test_gradients_reach_every_parameter(self):
        for pooling in ("last", "attn"):
            proj = self.projector(pooling, n_heads=2, d_attn=16)
            states = torch.randn(4, 8, self.D_IN, requires_grad=False)
            proj(states).pow(2).sum().backward()
            missing = [n for n, p in proj.named_parameters() if p.grad is None]
            assert not missing, missing

    def test_state_dict_and_pickle_round_trip(self):
        import pickle

        proj = self.projector("attn", n_heads=2, d_attn=16)
        clone = pickle.loads(pickle.dumps(proj))
        assert type(clone) is type(proj)
        states = torch.randn(2, 8, self.D_IN)
        assert torch.equal(clone(states), proj(states))
        other = self.projector("attn", n_heads=2, d_attn=16)
        other.load_state_dict({k: v + 1 for k, v in proj.state_dict().items()})
        assert not torch.equal(other(states), proj(states))

    def test_rejects_bad_arguments(self):
        from genetic_coil_arena.channels import LatentProjector

        with pytest.raises(ValueError, match="pooling"):
            LatentProjector(8, 8, 1, pooling="mean")
        with pytest.raises(ValueError):
            LatentProjector(8, 8, 0)

    def test_for_backend_uses_the_embedding_scale(self):
        from genetic_coil_arena.channels import LatentProjector

        class Stub:
            hidden_size = 16

            def embedding_rms(self):
                return 0.033

        proj = LatentProjector.for_backend(Stub(), 2)
        assert proj.d_in == proj.d_out == 16 and proj.k == 2
        assert float(proj.gain[0]) == pytest.approx(0.033)
