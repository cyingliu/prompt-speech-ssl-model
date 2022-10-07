import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from fairseq.modules import (SamePad)

def downsample(x, x_len, sample_rate, sample_style):
    batch_size, timestep, feature_dim = x.shape
    x_len = x_len // sample_rate

    if sample_style == 'drop':
        # Drop the unselected timesteps
        x = x[:, ::sample_rate, :].contiguous()
    elif sample_style == 'concat':
        # Drop the redundant frames and concat the rest according to sample rate
        if timestep % sample_rate != 0:
            x = x[:, :-(timestep % sample_rate), :]
        x = x.contiguous().view(batch_size, int(
            timestep / sample_rate), feature_dim * sample_rate)
    else:
        raise NotImplementedError
    
    return x, x_len


class RNNLayer(nn.Module):
    ''' RNN wrapper, includes time-downsampling'''

    def __init__(self, input_dim, module, bidirection, dim, dropout, layer_norm, sample_rate, proj):
        super(RNNLayer, self).__init__()
        # Setup
        rnn_out_dim = 2 * dim if bidirection else dim
        self.out_dim = rnn_out_dim
        self.dropout = dropout
        self.layer_norm = layer_norm
        self.sample_rate = sample_rate
        self.proj = proj

        # Recurrent layer
        self.layer = getattr(nn, module.upper())(
            input_dim, dim, bidirectional=bidirection, num_layers=1, batch_first=True)

        # Regularizations
        if self.layer_norm:
            self.ln = nn.LayerNorm(rnn_out_dim)
        if self.dropout > 0:
            self.dp = nn.Dropout(p=dropout)

        # Additional projection layer
        if self.proj:
            self.pj = nn.Linear(rnn_out_dim, rnn_out_dim)

    def forward(self, input_x, x_len):
        # Forward RNN
        if not self.training:
            self.layer.flatten_parameters()

        input_x = pack_padded_sequence(input_x, x_len, batch_first=True, enforce_sorted=False)
        output, _ = self.layer(input_x)
        output, x_len = pad_packed_sequence(output, batch_first=True)

        # Normalizations
        if self.layer_norm:
            output = self.ln(output)
        if self.dropout > 0:
            output = self.dp(output)

        # Perform Downsampling
        if self.sample_rate > 1:
            output, x_len = downsample(output, x_len, self.sample_rate, 'drop')

        if self.proj:
            output = torch.tanh(self.pj(output))

        return output, x_len


class RNNs(nn.Module):
    def __init__(self,
        input_size,
        output_size,
        upstream_rate,
        module,
        bidirection,
        dim,
        dropout,
        layer_norm,
        proj,
        sample_rate,
        sample_style,
        total_rate = 320,
    ):
        super(RNNs, self).__init__()
        latest_size = input_size

        self.sample_rate = 1 if total_rate == -1 else round(total_rate / upstream_rate)
        self.sample_style = sample_style
        if sample_style == 'concat':
            latest_size *= self.sample_rate

        self.rnns = nn.ModuleList()
        for i in range(len(dim)):
            rnn_layer = RNNLayer(
                latest_size,
                module,
                bidirection,
                dim[i],
                dropout[i],
                layer_norm[i],
                sample_rate[i],
                proj[i],
            )
            self.rnns.append(rnn_layer)
            latest_size = rnn_layer.out_dim

        self.linear = nn.Linear(latest_size, output_size)
    
    def forward(self, x, x_len):
        r"""
        Args:
            x (torch.Tensor): Tensor of dimension (batch_size, input_length, num_features).
            x_len (torch.IntTensor): Tensor of dimension (batch_size).
        Returns:
            Tensor: Predictor tensor of dimension (batch_size, input_length, number_of_classes).
        """
        # Perform Downsampling
        if self.sample_rate > 1:
            x, x_len = downsample(x, x_len, self.sample_rate, self.sample_style)
        for rnn in self.rnns:
            x, x_len = rnn(x, x_len)
        logits = self.linear(x)
        return logits, x_len        


class Wav2Letter(nn.Module):
    """
    The Wav2Letter model modified from torchaudio.models.Wav2Letter which preserves
    total downsample rate given the different upstream downsample rate.
    """

    def __init__(self, input_dim, output_dim, upstream_rate, total_rate=320, **kwargs):
        super(Wav2Letter, self).__init__()
        first_stride = 1 if total_rate == -1 else total_rate // upstream_rate
        self.downsample_rate = first_stride

        self.acoustic_model = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=250, kernel_size=48, stride=first_stride, padding=23),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=250, out_channels=250, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=250, out_channels=250, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=250, out_channels=250, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=250, out_channels=250, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=250, out_channels=250, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=250, out_channels=250, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=250, out_channels=250, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=250, out_channels=2000, kernel_size=32, stride=1, padding=16),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=2000, out_channels=2000, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=2000, out_channels=output_dim, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, x_len):
        r"""
        Args:
            x (torch.Tensor): Tensor of dimension (batch_size, input_length, num_features).
            x_len (torch.IntTensor): Tensor of dimension (batch_size).
        Returns:
            Tensor: Predictor tensor of dimension (batch_size, input_length, number_of_classes).
        """
        x = self.acoustic_model(x.transpose(1, 2).contiguous())
        return x.transpose(1, 2).contiguous(), x_len // self.downsample_rate

class Attentions(nn.Module):
    def __init__(self, input_dim, output_dim, upstream_rate, num_attention_head, num_layers, max_seq_len, positional_embedding_type, dropout):
        super(Attentions, self).__init__()
        print(f"embed_dim: {input_dim}, num_attention_head: {num_attention_head}, num_layers: {num_layers}")
        embed_dim = input_dim
        self.num_layers = num_layers
        self.attentions = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.positional_embedding_type = positional_embedding_type
        if positional_embedding_type == 'conv':
            self.wpe = nn.Sequential(
                nn.Conv1d(embed_dim, embed_dim, kernel_size=128, stride=1, padding=64, groups=16),
                SamePad(128),
                nn.GELU()
            )
        elif positional_embedding_type == 'embed':
            self.wpe = nn.Embedding(max_seq_len, embed_dim)
            # Ref: https://github.com/huggingface/transformers/blob/5ab21b072fa2a122da930386381d23f95de06e28/src/transformers/modeling_gpt2.py#L339-L342
            self.wpe.weight.data.normal_(mean=0.0, std=0.02)
        else:
            raise ValueError(f'Invalid positional_embedding_type: {positional_embedding_type}')
        
        for i in range(num_layers):
            attention_layer = nn.MultiheadAttention(embed_dim, num_attention_head, batch_first=True)
            self.attentions.append(attention_layer)
            self.dropouts.append(nn.Dropout(p=dropout[i]))

        
        self.linear = nn.Linear(embed_dim, output_dim)
        
           
    def forward(self, x, x_len):
        """
        x: (batch, seq_len, embed_dim)
        """
        bs = x.shape[0]
        device = x.device
        key_padding_mask = torch.zeros((bs, x.shape[1]), dtype=torch.bool)
        for i in range(bs):
            key_padding_mask[i, x_len[i]:] = True
        key_padding_mask = key_padding_mask.to(device)
        
        if self.positional_embedding_type == 'conv':
            positional_embeds = self.wpe(x.transpose(1, 2))
            positional_embeds = positional_embeds.transpose(1, 2)
        elif self.positional_embedding_type == 'embed':
            position_ids = torch.arange(0, x.shape[1], dtype=torch.long, device=x.device)
            position_ids = position_ids.unsqueeze(0).view(-1, x.shape[1])
            positional_embeds = self.wpe(position_ids)
        
        x = x + positional_embeds
        
        for i in range(self.num_layers):
            x, attn = self.attentions[i](
                query = x,
                key = x,
                value = x,
                key_padding_mask = key_padding_mask
            )
            x = self.dropouts[i](x)
       # x: (bs, seq_len, embed_dim)
        x = self.linear(x) # (bs, seq_len, output_dim)
        
        return x, x_len




        
