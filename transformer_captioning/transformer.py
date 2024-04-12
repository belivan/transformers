# Credit to the CS-231n course at Stanford, from which this assignment is adapted
import numpy as np
import copy
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

class AttentionLayer(nn.Module):

    def __init__(self, embed_dim, dropout=0.1):

        super().__init__()
        self.embed_dim = embed_dim
        # TODO: Initialize the following layers and parameters to perform attention
        # This class assumes that the input dimension for query, key and value is embed_dim
        self.query_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.key_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.value_proj = nn.Linear(self.embed_dim, self.embed_dim)

        self.dropout = nn.Dropout(dropout)  # dropout layer used in attention for regularization

    def forward(self, query, key, value, attn_mask=None):
        N, S, D = query.shape
        N, T, D = value.shape
        assert key.shape == value.shape

        # TODO : Compute attention 

        #project query, key and value  - 
        query = self.query_proj(query)
        key = self.key_proj(key)
        value = self.value_proj(value)

        #compute dot-product attention. Don't forget the scaling value!
        #Expected shape of dot_product is (N, S, T)
        dot_product = torch.bmm(query, key.transpose(1, 2)) / math.sqrt(D)

        if attn_mask is not None:
            # convert att_mask which is multiplicative, to an additive mask
            # Hint : If mask[i,j] = 0, we want softmax(QKT[i,j] + additive_mask[i,j]) to be 0
            # Think about what inputs make softmax 0.
            print(attn_mask.shape)
            additive_mask = attn_mask.unsqueeze(0).expand(N, -1, -1)
            dot_product += additive_mask

        # apply softmax, dropout, and use value
        y = self.dropout(F.softmax(dot_product + 1e-10, dim=2)) @ value
        return y

class MultiHeadAttentionLayer(AttentionLayer):

    def __init__(self, embed_dim, num_heads, dropout=0.1):
       
        super().__init__(embed_dim, dropout)
        self.num_heads = num_heads

        # TODO: Initialize the following layers and parameters to perform attention
        self.head_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, query, key, value, attn_mask=None):
        H = self.num_heads
        N, S, D = query.shape
        N, T, D = value.shape
        assert key.shape == value.shape

        # TODO : Compute multi-head attention
 
        #project query, key and value
        #after projection, split the embedding across num_heads
        #eg - expected shape for value is (N, H, T, D/H)
        query = self.query_proj(query).view(N, S, H, D // H).transpose(1, 2)
        key = self.key_proj(key).view(N, T, H, D // H).transpose(1, 2)
        value = self.value_proj(value).view(N, T, H, D // H).transpose(1, 2)

        #compute dot-product attention separately for each head. Don't forget the scaling value!
        #Expected shape of dot_product is (N, H, S, T)
        dot_product = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(D // H)

        if attn_mask is not None:
            # convert att_mask which is multiplicative, to an additive mask
            # Hint : If mask[i,j] = 0, we want softmax(QKT[i,j] + additive_mask[i,j]) to be 0
            # Think about what inputs make softmax 0.
            additive_mask = attn_mask.unsqueeze(0).unsqueeze(0).expand(N, H, -1, -1)
            additive_mask = additive_mask.to(dot_product.dtype).to(dot_product.device)
            dot_product += additive_mask

        # apply softmax, dropout, and use value
        y = self.dropout(F.softmax(dot_product + 1e-10, dim=3)) @ value

        # concat embeddings from different heads, and project
        output = self.head_proj(y.transpose(1, 2).contiguous().view(N, S, D))
        return output

        # print("Input shapes and sample values:")
        # print(f"  Query shape: {query.shape}, Sample: {query[0, 0, :5]}")
        # print(f"  Key shape: {key.shape}, Sample: {key[0, 0, :5]}")
        # print(f"  Value shape: {value.shape}, Sample: {value[0, 0, :5]}")

        # # Project query, key, and value
        # query = self.query_proj(query).view(N, S, H, D // H).transpose(1, 2)
        # key = self.key_proj(key).view(N, T, H, D // H).transpose(1, 2)
        # value = self.value_proj(value).view(N, T, H, D // H).transpose(1, 2)

        # print("After projection and reshaping:")
        # print(f"  Projected Query shape: {query.shape}, Sample: {query[0, 0, 0, :5]}")
        # print(f"  Projected Key shape: {key.shape}, Sample: {key[0, 0, 0, :5]}")
        # print(f"  Projected Value shape: {value.shape}, Sample: {value[0, 0, 0, :5]}")

        # # Compute dot-product attention separately for each head
        # dot_product = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(D // H)
        # print("Dot-product Shape and sample values:")
        # print(f"  Shape: {dot_product.shape}, Sample: {dot_product[0, 0, 0, :5]}")

        # if attn_mask is not None:
        #     print("Original Attention Mask Shape:", attn_mask.shape)
        #     additive_mask = attn_mask.unsqueeze(0).unsqueeze(0).expand(N, H, -1, -1)
        #     additive_mask = additive_mask.to(dot_product.dtype).to(dot_product.device)
        #     print("Additive Mask Shape and sample values:")
        #     print(f"  Shape: {additive_mask.shape}, Sample: {additive_mask[0, 0, 0, :5]}")
        #     dot_product += additive_mask

        # # Apply softmax, then dropout, and use the resulting attention scores to weight the values
        # attn_scores = F.softmax(dot_product + 1e-10, dim=3)
        # y = self.dropout(attn_scores) @ value
        # print("Attention Scores and weighted values:")
        # print(f"  Scores Shape: {attn_scores.shape}, Scores Sample: {attn_scores[0, 0, 0, :5]}")
        # print(f"  Weighted Value Shape: {y.shape}, Weighted Value Sample: {y[0, 0, 0, :5]}")

        # # Concatenate embeddings from different heads and project back to original dimension
        # output = self.head_proj(y.transpose(1, 2).contiguous().view(N, S, D))
        # print("Output Shape and sample values after head projection:")
        # print(f"  Shape: {output.shape}, Sample: {output[0, 0, :5]}")

        return output


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, dropout=0.1, max_len=5000):
        super().__init__()
        # TODO - use torch.nn.Embedding to create the encoding. Initialize dropout layer.
        self.encoding = nn.Embedding(max_len, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        N, S, D = x.shape
        # TODO - add the encoding to x

        output = x + self.encoding.weight[:S].unsqueeze(0)
        output = self.dropout(output)

        return output


class SelfAttentionBlock(nn.Module):

    def __init__(self, input_dim, num_heads, dropout=0.1):
        super().__init__()
        # TODO: Initialize the following. Use MultiHeadAttentionLayer for self_attn.
        self.self_attn = MultiHeadAttentionLayer(input_dim, num_heads, dropout)
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(input_dim)

    def forward(self, seq, mask):
        ############# TODO - Self-attention on the sequence, using the mask. Add dropout to attention layer output.
        # Then add a residual connection to the original input, and finally apply normalization. #############################
        x = self.self_attn(seq, seq, seq, mask)
        x = self.dropout(x)
        out = self.layernorm(seq + x)
        return out
        # print("Entering SelfAttentionBlock:")
        # print("  Input seq shape:", seq.shape)
        # print("  Mask shape:", mask.shape)
        # print("  Sample input seq (first element):", seq[0, 0, :5])

        # # Apply multi-head self-attention
        # attn_output = self.self_attn(seq, seq, seq, mask)
        # print("After MultiHeadAttention:")
        # print("  Attention output shape:", attn_output.shape)
        # print("  Sample attention output (first element):", attn_output[0, 0, :5])

        # # Apply dropout to the attention output
        # attn_output_dropout = self.dropout(attn_output)
        # print("After Dropout:")
        # print("  Dropout output shape:", attn_output_dropout.shape)
        # print("  Sample dropout output (first element):", attn_output_dropout[0, 0, :5])

        # # Add the original input sequence to the attention output to create a residual connection
        # residual_connection = seq + attn_output_dropout
        # print("After Adding Residual Connection:")
        # print("  Residual connection output shape:", residual_connection.shape)
        # print("  Sample residual output (first element):", residual_connection[0, 0, :5])

        # # Normalize the result using Layer Normalization
        # normalized_output = self.layernorm(residual_connection)
        # print("After Layer Normalization:")
        # print("  Normalized output shape:", normalized_output.shape)
        # print("  Sample normalized output (first element):", normalized_output[0, 0, :5])

        # return normalized_output

class CrossAttentionBlock(nn.Module):

    def __init__(self, input_dim, num_heads, dropout=0.1):
        super().__init__()
        # TODO: Initialize the following. Use MultiHeadAttentionLayer for cross_attn.
        self.cross_attn = MultiHeadAttentionLayer(input_dim, num_heads, dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, seq, cond):
        ############# TODO - Cross-attention on the sequence, using conditioning. Add dropout to attention layer output.
        # Then add a residual connection to the original input, and finally apply normalization. #############################
        x = self.cross_attn(seq, cond, cond)
        x = self.dropout(x)
        out = self.norm(seq + x)
        return out

class FeedForwardBlock(nn.Module):
    def __init__(self, input_dim, num_heads, dim_feedforward=2048, dropout=0.1 ):
        super().__init__()
        # TODO: Initialize the following. 
        # MLP has the following layers : linear, relu, dropout, linear ; hidden dim of linear is given by dim_feedforward
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, input_dim)
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, seq):
        ############# TODO - MLP on the sequence. Add dropout to mlp layer output.
        # Then add a residual connection to the original input, and finally apply normalization. #############################
        x = self.mlp(seq)
        x = self.dropout(x)
        out = self.norm(seq + x)
        return out

class DecoderLayer(nn.Module):
    def __init__(self, input_dim, num_heads, dim_feedforward=2048, dropout=0.1 ):
        super().__init__()
        self.self_atn_block = SelfAttentionBlock(input_dim, num_heads, dropout)
        self.cross_atn_block = CrossAttentionBlock(input_dim, num_heads, dropout)
        self.feedforward_block = FeedForwardBlock(input_dim, num_heads, dim_feedforward, dropout)

    def forward(self, seq, cond, mask):
        out = self.self_atn_block(seq, mask)
        out = self.cross_atn_block(out, cond)
        return self.feedforward_block(out)
        # print("Input to DecoderLayer:")
        # print("  seq shape:", seq.shape)  # Sequence input shape
        # print("  cond shape:", cond.shape)  # Conditioning input shape
        # print("  mask shape:", mask.shape)  # Mask shape
        # print("  Sample seq element:", seq[0, 0, :5])  # Sample from the sequence input
        # print("  Sample cond element:", cond[0, 0, :5])  # Sample from the conditioning input

        # out = self.self_atn_block(seq, mask)
        # print("Output after SelfAttentionBlock:")
        # print("  out shape:", out.shape)  # Output shape after self-attention
        # print("  Sample out element:", out[0, 0, :5])  # Sample from the output

        # out = self.cross_atn_block(out, cond)
        # print("Output after CrossAttentionBlock:")
        # print("  out shape:", out.shape)  # Output shape after cross-attention
        # print("  Sample out element:", out[0, 0, :5])  # Sample from the output

        # out = self.feedforward_block(out)
        # print("Output after FeedForwardBlock:")
        # print("  out shape:", out.shape)  # Output shape after feedforward block
        # print("  Sample out element:", out[0, 0, :5])  # Sample from the output

        # return out

class TransformerDecoder(nn.Module):
    def __init__(self, word_to_idx, idx_to_word, input_dim, embed_dim, num_heads=4,
                 num_layers=2, max_length=50, device = 'cuda'):
        """
        Construct a new TransformerDecoder instance.
        Inputs:
        - word_to_idx: A dictionary giving the vocabulary. It contains V entries.
          and maps each string to a unique integer in the range [0, V).
        - input_dim: Dimension of input image feature vectors.
        - embed_dim: Embedding dimension of the transformer.
        - num_heads: Number of attention heads.
        - num_layers: Number of transformer layers.
        - max_length: Max possible sequence length.
        """
        super().__init__()

        vocab_size = len(word_to_idx)
        self._null = word_to_idx["<NULL>"]
        self._start = word_to_idx.get("<START>", None)
        self.idx_to_word = idx_to_word

        self.layers = nn.ModuleList([DecoderLayer(embed_dim, num_heads) for _ in range(num_layers)])

        self.caption_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=self._null)
        self.positional_encoding = PositionalEncoding(embed_dim, max_len=max_length)
        self.feature_embedding = nn.Linear(input_dim, embed_dim)
        self.score_projection = nn.Linear(embed_dim, vocab_size) 

        self.apply(self._init_weights)
        self.device = device 
        self.to(device)

    def get_data_embeddings(self, features, captions):
        # TODO - get caption and feature embeddings 
        # Don't forget position embeddings for captions!
        # expected caption embedding output shape : (N, T, D)
        caption_embedding = self.caption_embedding(captions)
        caption_embedding = self.positional_encoding(caption_embedding)

        # Unsqueeze feature embedding along dimension 1
        # expected feature embedding output shape : (N, 1, D)
        feature_embedding = self.feature_embedding(features).unsqueeze(1)
        return feature_embedding, caption_embedding

    def get_causal_mask(self, _len):
        # TODO - get causal mask. This should be a matrix of shape (_len, _len). 
        # This mask is multiplicative
        # setting mask[i,j] = 0 means jth element of the sequence is not used 
        # to predict the ith element of the sequence.
        mask = torch.triu(torch.ones(_len, _len) * float('-inf'), diagonal=1)
        return mask

    def forward(self, features, captions):
        """
        Given image features and caption tokens, return a distribution over the
        possible tokens for each timestep. Note that since the entire sequence
        of captions is provided all at once, we mask out future timesteps.
        Inputs:
         - features: image features, of shape (N, D)
         - captions: ground truth captions, of shape (N, T)
        Returns:
         - scores: score for each token at each timestep, of shape (N, T, V)
        """
        features_embed, captions_embed = self.get_data_embeddings(features, captions)
        mask = self.get_causal_mask(captions_embed.shape[1])
        mask.to(captions_embed.dtype)

        output = captions_embed
        for layer in self.layers:
            output = layer(output, features_embed, mask=mask)

        scores = self.score_projection(output)
        return scores
        # features_embed, captions_embed = self.get_data_embeddings(features, captions)
        # print("After embedding")
        # print("Features Embed shape:", features_embed.shape)  # Should be (N, 1, D)
        # print("Captions Embed shape:", captions_embed.shape)  # Should be (N, T, D)
        # print("Sample Feature Embed:", features_embed[0, 0, :5])  # Print first 5 dims of the first feature embedding
        # print("Sample Caption Embed:", captions_embed[0, 0, :5])  # Print first 5 dims of the first caption embedding

        # # Getting causal mask to prevent attention to future timesteps
        # mask = self.get_causal_mask(captions_embed.shape[1])
        # mask = mask.to(captions_embed.device)  # Ensuring mask is on the correct device
        # print("Mask shape:", mask.shape)  # Should be (T, T)
        # print("Mask sample:", mask[:, :5])  # Print first 5 columns of the mask

        # output = captions_embed
        # for layer in self.layers:
        #     output = layer(output, features_embed, mask=mask)
        #     print("After layer output shape:", output.shape)  # Should be (N, T, D)
        #     print("Sample Output (first element of the batch):", output[0, 0, :5])

        # scores = self.score_projection(output)
        # print("Scores shape:", scores.shape)  # Should be (N, T, V)
        # print("Sample Scores:", scores[0, 0, :5])  # Print first 5 score dims for the first token

        # return scores

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def sample(self, features, max_length=30):
        """
        Given image features, use greedy decoding to predict the image caption.
        Inputs:
         - features: image features, of shape (N, D)
         - max_length: maximum possible caption length
        Returns:
         - captions: captions for each example, of shape (N, max_length)
        """
        with torch.no_grad():
            features = torch.Tensor(features).to(self.device)
            N = features.shape[0]

            # Create an empty captions tensor (where all tokens are NULL).
            captions = self._null * np.ones((N, max_length), dtype=np.int32)

            # Create a partial caption, with only the start token.
            partial_caption = self._start * np.ones(N, dtype=np.int32)
            partial_caption = torch.LongTensor(partial_caption).to(self.device)
            # [N] -> [N, 1]
            partial_caption = partial_caption.unsqueeze(1)

            for t in range(max_length):

                # Predict the next token (ignoring all other time steps).
                output_logits = self.forward(features, partial_caption)
                output_logits = output_logits[:, -1, :]

                # Choose the most likely word ID from the vocabulary.
                # [N, V] -> [N]
                word = torch.argmax(output_logits, axis=1)

                # Update our overall caption and our current partial caption.
                captions[:, t] = word.cpu().numpy()
                word = word.unsqueeze(1)
                partial_caption = torch.cat([partial_caption, word], dim=1)

            return captions
