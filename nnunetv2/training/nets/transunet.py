from typing import Union, Type, List, Tuple

import torch
from dynamic_network_architectures.architectures.abstract_arch import (
    test_submodules_loadable,
)
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim


from dynamic_network_architectures.architectures.unet import PlainConvUNet
from dynamic_network_architectures.initialization.weight_init import InitWeights_He

from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd


class TransUNet(PlainConvUNet):
    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op: Type[_ConvNd],
        kernel_sizes: Union[int, List[int], Tuple[int, ...]],
        strides: Union[int, List[int], Tuple[int, ...]],
        n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
        num_classes: int,
        n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
        conv_bias: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict = None,  # type: ignore
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: dict = None,  # type: ignore
        nonlin: Union[None, Type[torch.nn.Module]] = None,
        nonlin_kwargs: dict = None,  # type: ignore
        deep_supervision: bool = False,
        nonlin_first: bool = False,
        n_heads: int = 8,
        dropout_rate: float = 0.1,
        positional_encoding: bool = True,
        num_transformer_blocks: int = 4,
    ):
        """
        nonlin_first: if True you get conv -> nonlin -> norm. Else it's conv -> norm -> nonlin
        """
        super().__init__(
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=conv_op,
            input_channels=input_channels,
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_conv_per_stage=n_conv_per_stage,
            num_classes=num_classes,
            n_conv_per_stage_decoder=n_conv_per_stage_decoder,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            dropout_op=dropout_op,
            dropout_op_kwargs=dropout_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            deep_supervision=deep_supervision,
            nonlin_first=nonlin_first,
        )

        # 添加Transformer组件
        # 获取最后一个stage的特征维度
        last_stage_features = (
            features_per_stage[-1]
            if isinstance(features_per_stage, (list, tuple))
            else features_per_stage
        )

        # 保存位置编码选项
        self.positional_encoding = positional_encoding

        # 创建位置编码层（如果启用）
        if self.positional_encoding:
            self.positional_encoding_layer = self._make_positional_encoding(
                last_stage_features
            )

        self.transformer_block = self._make_transformer_block(
            last_stage_features,
            n_heads,
            dropout_rate,
            num_layers=num_transformer_blocks,
        )

    def _make_transformer_block(self, hidden_dim, n_heads, dropout_rate, num_layers=1):
        """创建Transformer编码器块"""
        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dropout=dropout_rate,
            batch_first=True,  # 支持(batch, seq, feature)格式
        )
        # Transformer encoder
        transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        return transformer

    def _make_positional_encoding(self, hidden_dim, max_len=4096):
        """创建位置编码层"""
        # 创建可学习的位置编码
        positional_encoding = nn.Parameter(torch.zeros(1, max_len, hidden_dim))
        nn.init.normal_(positional_encoding, mean=0, std=0.02)
        return positional_encoding

    def forward(self, x):
        skips = self.encoder(x)

        # 对skip连接中的最后一个特征图应用Transformer
        # 注意：我们需要将特征图从(N, C, H, W)转换为(N, H*W, C)格式以适应Transformer
        last_skip = skips[-1]  # 获取最后一个skip连接 (N, C, H, W)
        batch_size, channels, height, width = last_skip.shape

        # 重塑为序列格式 (N, H*W, C)
        last_skip_reshaped = last_skip.view(batch_size, channels, -1).permute(0, 2, 1)

        # 添加位置编码（如果启用）
        if self.positional_encoding:
            # 截取与序列长度相匹配的位置编码
            positional_encoding = self.positional_encoding_layer[
                :, : last_skip_reshaped.size(1), :
            ]
            last_skip_reshaped = last_skip_reshaped + positional_encoding

        # 应用Transformer
        transformed_skip = self.transformer_block(last_skip_reshaped)

        # 转换回原始形状 (N, C, H, W)
        transformed_skip = transformed_skip.permute(0, 2, 1).view(
            batch_size, channels, height, width
        )

        # 替换skip连接中的最后一个特征图
        skips[-1] = transformed_skip

        return self.decoder(skips)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), (
            "just give the image size without color/feature channels or "
            "batch channel. Do not give input_size=(b, c, x, y(, z)). "
            "Give input_size=(x, y(, z))!"
        )
        return self.encoder.compute_conv_feature_map_size(
            input_size
        ) + self.decoder.compute_conv_feature_map_size(input_size)

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)


if __name__ == "__main__":

    data = torch.rand((1, 3, 512, 512))

    model = TransUNet(
        3,
        7,
        (32, 64, 125, 256, 512, 512, 512),
        nn.Conv2d,
        3,
        (1, 2, 2, 2, 2, 2, 2),
        (2, 2, 2, 2, 2, 2, 2),
        2,
        (2, 2, 2, 2, 2, 2),
        False,
        nn.BatchNorm2d,
        None,
        None,
        None,
        nn.ReLU,
        deep_supervision=True,
        num_transformer_blocks=4,
    )
    output = model(data)[0]
    print("Output shape:", output.shape)

    test_submodules_loadable(model)
