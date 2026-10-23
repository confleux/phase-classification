import pytest
import torch

from phase_classification.models.temporal_adapter import ResidualTemporalAdapter


def test_residual_temporal_adapter_preserves_shape_and_starts_as_identity():
    adapter = ResidualTemporalAdapter(feature_dim=12, bottleneck_dim=4)
    features = torch.randn(2, 7, 12)

    output = adapter(features)

    assert output.shape == features.shape
    torch.testing.assert_close(output, features)


def test_residual_temporal_adapter_learns_a_temporal_residual():
    adapter = ResidualTemporalAdapter(feature_dim=8, bottleneck_dim=4)
    features = torch.randn(2, 5, 8)
    downstream = torch.nn.Linear(8, 3)

    downstream(adapter(features)).square().mean().backward()

    assert adapter.up.weight.grad is not None
    assert adapter.up.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("shape", [(5, 8), (2, 5, 7)])
def test_residual_temporal_adapter_rejects_invalid_feature_shapes(shape):
    adapter = ResidualTemporalAdapter(feature_dim=8, bottleneck_dim=4)

    with pytest.raises(ValueError, match="expects features shaped"):
        adapter(torch.randn(*shape))


def test_residual_temporal_adapter_rejects_even_kernel_size():
    with pytest.raises(ValueError, match="positive odd integer"):
        ResidualTemporalAdapter(feature_dim=8, kernel_size=2)
