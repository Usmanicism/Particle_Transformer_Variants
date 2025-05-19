from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn, Tensor


def _get_relative_position_index(height: int, width: int) -> torch.Tensor:
    coords = torch.stack(torch.meshgrid([torch.arange(height), torch.arange(width)]))
    coords_flat = torch.flatten(coords, 1)
    relative_coords = coords_flat[:, :, None] - coords_flat[:, None, :]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()
    relative_coords[:, :, 0] += height - 1
    relative_coords[:, :, 1] += width - 1
    relative_coords[:, :, 0] *= 2 * width - 1
    return relative_coords.sum(-1)


class RelativePositionalMultiHeadAttention(nn.Module):
    """Relative Positional Multi-Head Attention.

    Args:
        feat_dim (int): Number of input features.
        head_dim (int): Number of features per head.
        max_seq_len (int): Maximum sequence length.
    """

    def __init__(
        self,
        feat_dim: int,
        head_dim: int,
        max_seq_len: int,
    ) -> None:
        super().__init__()

        if feat_dim % head_dim != 0:
            raise ValueError(f"feat_dim: {feat_dim} must be divisible by head_dim: {head_dim}")

        self.n_heads = feat_dim // head_dim
        self.head_dim = head_dim
        self.size = int(max_seq_len)
        self.max_seq_len = max_seq_len

        self.to_qkv = nn.Linear(feat_dim, self.n_heads * self.head_dim * 3)
        self.scale_factor = feat_dim**-0.5

        self.merge = nn.Linear(self.head_dim * self.n_heads, feat_dim)
        self.relative_position_bias_table = nn.parameter.Parameter(
            torch.empty(((2 * self.size - 1), self.n_heads), dtype=torch.float32),
        )

        self.register_buffer("relative_position_index", _get_relative_position_index(1, self.size))
        # initialize with truncated normal the bias
        torch.nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def get_relative_positional_bias(self) -> torch.Tensor:
        bias_index = self.relative_position_index.view(-1)  # type: ignore
        relative_bias = self.relative_position_bias_table[bias_index].view(self.max_seq_len, self.max_seq_len, -1)  # type: ignore
        relative_bias = relative_bias.permute(2, 0, 1).contiguous()
        return relative_bias.unsqueeze(0)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Input tensor with expected layout of [B, G, P, D].
        Returns:
            Tensor: Output tensor with expected layout of [B, G, P, D].
        """
        B, G, P, D = x.shape
        H, DH = self.n_heads, self.head_dim

        qkv = self.to_qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        q = q.reshape(B, G, P, H, DH).permute(0, 1, 3, 2, 4)
        k = k.reshape(B, G, P, H, DH).permute(0, 1, 3, 2, 4)
        v = v.reshape(B, G, P, H, DH).permute(0, 1, 3, 2, 4)

        k = k * self.scale_factor
        dot_prod = torch.einsum("B G H I D, B G H J D -> B G H I J", q, k)
        pos_bias = self.get_relative_positional_bias()

        dot_prod = F.softmax(dot_prod + pos_bias, dim=-1)

        out = torch.einsum("B G H I J, B G H J D -> B G H I D", dot_prod, v)
        out = out.permute(0, 1, 3, 2, 4).reshape(B, G, P, D)

        out = self.merge(out)
        return out


class SwapAxes(nn.Module):
    """Permute the axes of a tensor."""

    def __init__(self, a: int, b: int) -> None:
        super().__init__()
        self.a = a
        self.b = b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = torch.swapaxes(x, self.a, self.b)
        return res


class WindowPartition(nn.Module):
    """
    Partition the input tensor into non-overlapping windows.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor, p: int) -> Tensor:
        """
        Args:
            x (Tensor): Input tensor with expected layout of [B, C, H, W] -> [B, C, L]
            p (int): Number of partitions.
        Returns:
            Tensor: Output tensor with expected layout of [B, H/P*W/P, P*P, C] -> [B, L/P, P, C]
        """
        # B, C, H, W = x.shape
        B, C, L = x.shape
        P = p
        # chunk up H and W dimensions
        # x = x.reshape(B, C, H // P, P, W // P, P)
        x = x.reshape(B, C, L // P, P)
        # x = x.permute(0, 2, 4, 3, 5, 1)
        x = x.permute(0, 2, 3, 1)
        # colapse P * P dimension
        # x = x.reshape(B, (H // P) * (W // P), P * P, C)
        return x


class WindowDepartition(nn.Module):
    """
    Departition the input tensor of non-overlapping windows into a feature volume of layout [B, C, H, W] -> [B, C, L]
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor, p: int, l_partitions: int) -> Tensor:
        """
        Args:
            x (Tensor): Input tensor with expected layout of [B, (H/P * W/P), P*P, C] -> [B, L/P, P, C]
            p (int): Number of partitions.
            l_partitions (int): Number of length partitions.
        Returns:
            Tensor: Output tensor with expected layout of [B, C, H, W] -> [B, C, L]
        """
        B, G, PP, C = x.shape
        P = p
        L = l_partitions
        # split P * P dimension into 2 P tile dimensionsa
        # x = x.reshape(B, HP, WP, P, P, C)
        # permute into B, C, HP, P, WP, P
        # x = x.permute(0, 5, 1, 3, 2, 4)
        x = x.permute(0, 3, 1, 2)
        # reshape into B, C, H, W
        # x = x.reshape(B, C, HP * P, WP * P)
        x = x.reshape(B, C, L * P)
        return x


class PartitionAttentionLayer(nn.Module):
    """
    Layer for partitioning the input tensor into non-overlapping windows and applying attention to each window.

    Args:
        in_channels (int): Number of input channels.
        head_dim (int): Dimension of each attention head.
        partition_size (int): Size of the partitions.
        partition_type (str): Type of partitioning to use. Can be either "grid" or "window".
        grid_size (Tuple[int, int]): Size of the grid to partition the input tensor into.
        mlp_ratio (int): Ratio of the  feature size expansion in the MLP layer.
        activation_layer (Callable[..., nn.Module]): Activation function to use.
        norm_layer (Callable[..., nn.Module]): Normalization function to use.
        attention_dropout (float): Dropout probability for the attention layer.
        mlp_dropout (float): Dropout probability for the MLP layer.
    """

    def __init__(
        self,
        in_channels: int,
        head_dim: int,
        # partitioning parameters
        partition_size: int,
        partition_type: str,
        # grid size needs to be known at initialization time
        # because we need to know hamy relative offsets there are in the grid
        grid_size: int,
        mlp_ratio: int,
        activation_layer: Callable[..., nn.Module],
        norm_layer: Callable[..., nn.Module],
        attention_dropout: float,
        mlp_dropout: float,
    ) -> None:
        super().__init__()

        self.n_heads = in_channels // head_dim
        self.head_dim = head_dim
        self.n_partitions = grid_size // partition_size
        self.partition_type = partition_type
        self.grid_size = grid_size

        if partition_type not in ["grid", "window"]:
            raise ValueError("partition_type must be either 'grid' or 'window'")

        if partition_type == "window":
            self.p, self.g = partition_size, self.n_partitions
        else:
            self.p, self.g = self.n_partitions, partition_size

        self.partition_op = WindowPartition()
        self.departition_op = WindowDepartition()
        self.partition_swap = SwapAxes(-2, -3) if partition_type == "grid" else nn.Identity()
        self.departition_swap = SwapAxes(-2, -3) if partition_type == "grid" else nn.Identity()

        self.attn_layer = nn.Sequential(
            norm_layer(in_channels),
            # it's always going to be partition_size ** 2 because
            # of the axis swap in the case of grid partitioning
            RelativePositionalMultiHeadAttention(in_channels, head_dim, partition_size),
            nn.Dropout(attention_dropout),
        )

        # pre-normalization similar to transformer layers
        self.mlp_layer = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, in_channels * mlp_ratio),
            activation_layer(),
            nn.Linear(in_channels * mlp_ratio, in_channels),
            nn.Dropout(mlp_dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Input tensor with expected layout of [B, C, H, W] -> [B, C, L]
        Returns:
            Tensor: Output tensor with expected layout of [B, C, H, W] -> [B, C, L]
        """

        # Undefined behavior if H or W are not divisible by p
        # https://github.com/google-research/maxvit/blob/da76cf0d8a6ec668cc31b399c4126186da7da944/maxvit/models/maxvit.py#L766
        g = self.grid_size // self.p
        torch._assert(
            self.grid_size % self.p == 0,
            "Grid size must be divisible by partition size. Got grid size of {} and partition size of {}".format(
                self.grid_size, self.p
            ),
        )

        print('before partiton x: ', x.shape)
        x = self.partition_op(x, self.p)
        print('after partiton x: ', x.shape)
        x = self.partition_swap(x)
        print('after swap x: ', x.shape)
        x = x + self.attn_layer(x)
        print('after attention x: ', x.shape)
        x = x + self.mlp_layer(x)
        print('after mlp x: ', x.shape)
        x = self.departition_swap(x)
        print('after departition swap x: ', x.shape)
        x = self.departition_op(x, self.p, g)
        print('after departition x: ', x.shape)

        return x


if __name__ == '__main__':
    window_attention = PartitionAttentionLayer(
        in_channels=128, 
        head_dim=8, 
        partition_size=8, 
        partition_type='window', 
        grid_size=128, 
        mlp_ratio=4, 
        activation_layer=nn.GELU, 
        norm_layer=nn.LayerNorm, 
        attention_dropout=0.1, 
        mlp_dropout=0.1)
    x = torch.rand((2, 128, 128))
    print('[x]', x.shape)
    y = window_attention(x)
    print('[y]', y.shape)

    grid_attention = PartitionAttentionLayer(
        in_channels=128, 
        head_dim=8, 
        partition_size=8, 
        partition_type='grid', 
        grid_size=128, 
        mlp_ratio=4, 
        activation_layer=nn.GELU, 
        norm_layer=nn.LayerNorm, 
        attention_dropout=0.1, 
        mlp_dropout=0.1)
    x = torch.rand((2, 128, 128))
    print('[x]', x.shape)
    y = grid_attention(x)
    print('[y]', y.shape)
