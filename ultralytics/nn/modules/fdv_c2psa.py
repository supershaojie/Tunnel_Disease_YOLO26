"""FDV-C2PSA v1: preserve native attention and add a bounded per-channel value high-pass residual."""

from torch import nn

from .block import C2PSA, Attention

__all__ = ("FDV_C2PSA", "FDVAttention")


class FDVAttention(Attention):
    """Take ownership of native attention tensors without reinitialization or changing their state paths.

    Args:
        native (Attention): Existing native attention whose QKV, PE and projection are retained.
    """

    def __init__(self, native: Attention):
        """Add only zero-initialized channel coefficients and a fixed, shape-preserving average pool."""
        nn.Module.__init__(self)
        self.num_heads, self.head_dim = native.num_heads, native.head_dim
        self.key_dim, self.scale = native.key_dim, native.scale
        self.qkv, self.proj, self.pe = native.qkv, native.proj, native.pe
        self.theta_c = nn.Parameter(native.qkv.conv.weight.new_zeros(self.num_heads * self.head_dim))
        self.lowpass = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

    @property
    def gamma_c(self):
        """Return the fixed 0.10*tanh(theta_c) coefficient for each value channel."""
        return 0.10 * self.theta_c.float().tanh()

    def forward(self, x):
        """Keep native Q/K, score, softmax, V aggregation and PE(V); add high-pass detail before proj."""
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )
        attn = (q * self.scale).transpose(-2, -1) @ k
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        spatial = v.reshape(B, C, H, W)
        high = spatial - self.lowpass(spatial)
        x = x + self.gamma_c.to(v.dtype).view(1, C, 1, 1) * high
        return self.proj(x)


class FDV_C2PSA(C2PSA):
    """Reuse native C2PSA and PSABlocks verbatim, replacing only each internal Attention."""

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        """Construct native parameters once; transferring attention consumes no additional RNG."""
        super().__init__(c1, c2, n, e)
        for block in self.m:
            block.attn = FDVAttention(block.attn)
