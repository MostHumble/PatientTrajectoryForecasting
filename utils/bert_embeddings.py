from typing import Optional
import torch.nn as nn
import torch
from utils.bert_layers_mosa import BertModel
from transformers import BertPreTrainedModel
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
import logging

logger = logging.getLogger(__name__)

class MosaicBertForEmbeddingGeneration(BertPreTrainedModel):
    """Bert Model transformer with a sequence classification/regression head.

    This head is just a linear layer on top of the pooled output.
    """

    def __init__(self, config):

        assert config.num_hidden_layers >= config.num_embedding_layers, 'num_hidden_layers should be greater than or equal to num_embedding_layers'
        
        self.config = config
        self.bert = BertModel(config)
       # this resets the weights
        self.post_init()


    @classmethod
    def from_pretrained(cls,
                      pretrained_checkpoint,
                      state_dict=None,
                      config=None,
                      *inputs,
                      **kwargs):
        """Load from pre-trained."""
        # this gets a fresh init model
        model = cls(config, *inputs, **kwargs)
        
        # thus we need to load the state_dict
        state_dict = torch.load(pretrained_checkpoint)
        # remove `model` prefix to avoid error
        consume_prefix_in_state_dict_if_present(state_dict, prefix='model.')
        missing_keys, unexpected_keys = model.load_state_dict(state_dict,
                                                              strict=False)

        if len(missing_keys) > 0:
            logger.warning(
                f"Found these missing keys in the checkpoint: {', '.join(missing_keys)}")

            logger.warning(f"the number of which is equal to {len(missing_keys)}"
            )

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
        return_dict: Optional[bool] = None,
        subset_mask : Optional[torch.Tensor] = None,
        masked_tokens_mask: Optional[torch.Tensor] = None,
        hospital_ids_lens: list = None,
    ) -> torch.Tensor:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if masked_tokens_mask is None:
            subset_mask = None

        embedding_output = self.bert.embeddings(input_ids, token_type_ids,
                                           position_ids)
        
        encoder_outputs_all = self.bert.encoder(
            embedding_output,
            attention_mask,
            output_all_encoded_layers=True,
            subset_mask=subset_mask)
        
        return self.get_embeddings(encoder_outputs_all, hospital_ids_lens, self.config.num_embedding_layers)
     
    def get_embeddings(self, encoder_outputs_all, hospital_ids_lens, num_layers):
        # num_layer (we use 4), batch_size (concatenated visits), seq_len (clinical note sequences), hidden_dim.
        sentence_representation = torch.stack(encoder_outputs_all[-num_layers:]).mean(dim=[0, 2])
        batch_embeddings = []
        start_idx = 0
        for length in hospital_ids_lens:
            # We then average across visits
            batch_embeddings.append(torch.mean(sentence_representation[start_idx:start_idx + length],dim=0))
            start_idx += length
        return torch.stack(batch_embeddings)