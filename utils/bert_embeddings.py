import logging
from typing import Optional

import torch
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from transformers import BertPreTrainedModel

from utils.bert_layers_mosa import BertModel

from transformers import PreTrainedModel

# Set up logging
logger = logging.getLogger(__name__)


class MosaicBertForEmbeddingGeneration(BertPreTrainedModel):

    def __init__(self, config, add_pooling_layer=False):
        """
        Initializes the BertEmbeddings class.

        Args:
            config (BertConfig): The configuration for the BERT model.
            add_pooling_layer (bool, optional): Whether to add a pooling layer. Defaults to False.
        """
        super().__init__(config)
        assert (
            config.num_hidden_layers >= config.num_embedding_layers
        ), "num_hidden_layers should be greater than or equal to num_embedding_layers"
        self.config = config
        self.strategy = config.strategy
        self.bert = BertModel(config, add_pooling_layer=add_pooling_layer)
        # this resets the weights
        self.post_init()

    @classmethod
    def from_pretrained(
        cls, pretrained_checkpoint, state_dict=None, config=None, *inputs, **kwargs
    ):
        """Load from pre-trained."""
        # this gets a fresh init model
        model = cls(config, *inputs, **kwargs)

        # thus we need to load the state_dict
        state_dict = torch.load(pretrained_checkpoint)
        # remove `model` prefix to avoid error
        consume_prefix_in_state_dict_if_present(state_dict, prefix="model.")
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        if len(missing_keys) > 0:
            logger.warning(
                f"Found these missing keys in the checkpoint: {', '.join(missing_keys)}"
            )

            logger.warning(f"the number of which is equal to {len(missing_keys)}")

        if len(unexpected_keys) > 0:
            logger.warning(
                f"Found these unexpected keys in the checkpoint: {', '.join(unexpected_keys)}",
            )
            logger.warning(f"the number of which is equal to {len(unexpected_keys)}")

        return model

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        subset_mask: Optional[torch.Tensor] = None,
        hospital_ids_lens: list = None,
    ) -> torch.Tensor:

        embedding_output = self.bert.embeddings(input_ids, token_type_ids, position_ids)

        encoder_outputs_all = self.bert.encoder(
            embedding_output,
            attention_mask,
            output_all_encoded_layers=True,
            subset_mask=subset_mask,
        )

        # batch_size, hidden_dim
        return self.get_embeddings(
            encoder_outputs_all,
            hospital_ids_lens,
            self.config.num_embedding_layers,
            self.config.strategy,
        )

    def get_embeddings(
        self, encoder_outputs_all, hospital_ids_lens, num_layers, strategy
    ):

        batch_embeddings = []
        start_idx = 0

        # num_layer (we use default = 4), batch_size (concatenated visits), seq_len (clinical note sequences), hidden_dim.
        # average across num_layers and seq_len
        if strategy == "mean":
            # batch_size (concatenated visits), hidden_dim.
            sentence_representation = torch.stack(
                encoder_outputs_all[-num_layers:]
            ).mean(dim=[0, 2])

            for length in hospital_ids_lens:
                # We then average across visits
                # batch_size (true batch size), hidden_dim.
                batch_embeddings.append(
                    torch.mean(
                        sentence_representation[start_idx : start_idx + length], dim=0
                    )
                )
                start_idx += length

            return torch.stack(batch_embeddings)

        elif strategy == "concat":
            # num_layer, batch_size (concatenated visits), hidden_dim.
            sentence_representation = torch.stack(
                encoder_outputs_all[-num_layers:]
            ).mean(dim=2)

            for length in hospital_ids_lens:
                # We then average across visits
                # num_layer, batch_size (true batch size), hidden_dim.
                batch_embeddings.append(
                    torch.mean(
                        sentence_representation[:, start_idx : start_idx + length],
                        dim=1,
                    )
                )
                start_idx += length

            return torch.stack(batch_embeddings)

        elif strategy == "all":
            # num_layer, batch_size (concatenated visits), seq_len (clinical note sequences), hidden_dim.
            sentence_representation = torch.stack(encoder_outputs_all[-num_layers:])
            return sentence_representation
        else:
            raise ValueError(
                f"{strategy} is not a valid strategy, choose between mean and concat"
            )

class MosaicBertForEmbeddingGenerationHF(PreTrainedModel):
    """A class to generate embeddings using the ClinicalMosaic model with customizable layers and strategies."""

    def __init__(self, pretrained_model, num_embedding_layers: int = 4, strategy: str = "mean"):
        """
        Initialize with a pre-loaded ClinicalMosaic model.

        Args:
            pretrained_model (PreTrainedModel): The pre-loaded ClinicalMosaic model.
            num_embedding_layers (int): Number of encoder layers to use for embeddings.
            strategy (str): Strategy for embedding generation ('mean', 'concat', 'all').
        """
        super().__init__(pretrained_model.config)
        self.bert = pretrained_model
        self.config = pretrained_model.config
        self.num_embedding_layers = num_embedding_layers
        self.strategy = strategy
        
        # Validate inputs
        assert self.config.num_hidden_layers >= self.num_embedding_layers, (
            f"num_embedding_layers ({self.num_embedding_layers}) must be <= num_hidden_layers ({self.config.num_hidden_layers})"
        )
        valid_strategies = {"mean", "concat", "all"}
        assert self.strategy in valid_strategies, f"Strategy must be one of {valid_strategies}"

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        hospital_ids_lens: Optional[list] = None,
    ) -> torch.Tensor:
        """
        Forward pass to generate embeddings.

        Args:
            input_ids (torch.Tensor, optional): Input token IDs.
            attention_mask (torch.Tensor, optional): Attention mask.
            token_type_ids (torch.Tensor, optional): Token type IDs.
            position_ids (torch.Tensor, optional): Position IDs.
            hospital_ids_lens (list, optional): Lengths for batch segmentation.

        Returns:
            torch.Tensor: Generated embeddings.
        """
        # Get all encoder layer outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            output_all_encoded_layers=True,
        )
        encoder_outputs_all = outputs  # Tuple of 12 tensors, each [batch_size, seq_len, hidden_size]

        # Generate embeddings using specified strategy
        return self.get_embeddings(
            encoder_outputs_all,
            hospital_ids_lens,
            self.num_embedding_layers,
            self.strategy,
        )

    def get_embeddings(
        self,
        encoder_outputs_all: tuple,
        hospital_ids_lens: Optional[list],
        num_layers: int,
        strategy: str,
    ) -> torch.Tensor:
        """
        Generate embeddings from encoder outputs.

        Args:
            encoder_outputs_all (tuple): Tuple of encoder layer outputs.
            hospital_ids_lens (list, optional): Lengths for batch segmentation.
            num_layers (int): Number of layers to use.
            strategy (str): Embedding strategy.

        Returns:
            torch.Tensor: Embeddings based on the strategy.
        """
        batch_embeddings = []
        start_idx = 0

        if strategy == "mean":
            # Stack last num_layers and average over layers and sequence
            sentence_representation = torch.stack(encoder_outputs_all[-num_layers:]).mean(dim=[0, 2])
            if hospital_ids_lens:
                for length in hospital_ids_lens:
                    batch_embeddings.append(
                        torch.mean(sentence_representation[start_idx:start_idx + length], dim=0)
                    )
                    start_idx += length
                return torch.stack(batch_embeddings)
            return sentence_representation

        elif strategy == "concat":
            # Stack last num_layers, average over sequence, keep layer dimension
            sentence_representation = torch.stack(encoder_outputs_all[-num_layers:]).mean(dim=2)
            if hospital_ids_lens:
                for length in hospital_ids_lens:
                    batch_embeddings.append(
                        torch.mean(sentence_representation[:, start_idx:start_idx + length], dim=1)
                    )
                    start_idx += length
                return torch.stack(batch_embeddings)
            return sentence_representation.view(sentence_representation.size(1), -1)

        elif strategy == "all":
            # Return all specified layers
            sentence_representation = torch.stack(encoder_outputs_all[-num_layers:])
            if hospital_ids_lens:
                logger.warning("hospital_ids_lens ignored with strategy 'all'")
            return sentence_representation

        else:
            raise ValueError(f"Invalid strategy: {strategy}. Choose 'mean', 'concat', or 'all'")