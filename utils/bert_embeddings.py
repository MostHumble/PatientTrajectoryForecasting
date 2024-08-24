import logging
from typing import Optional

import torch
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from transformers import BertPreTrainedModel

from utils.bert_layers_mosa import BertModel

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
        assert config.num_hidden_layers >= config.num_embedding_layers, 'num_hidden_layers should be greater than or equal to num_embedding_layers'
        self.config = config
        self.strategy = config.strategy
        self.bert = BertModel(config, add_pooling_layer=add_pooling_layer)
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
        subset_mask : Optional[torch.Tensor] = None,
        hospital_ids_lens: list = None,
    ) -> torch.Tensor:
        
        embedding_output = self.bert.embeddings(input_ids, token_type_ids,
                                           position_ids)
        
        encoder_outputs_all = self.bert.encoder(
            embedding_output,
            attention_mask,
            output_all_encoded_layers=True,
            subset_mask=subset_mask)
        
        # batch_size, hidden_dim
        return self.get_embeddings(encoder_outputs_all, hospital_ids_lens, self.config.num_embedding_layers, self.config.strategy)
     
    def get_embeddings(self, encoder_outputs_all, hospital_ids_lens, num_layers, strategy):

        batch_embeddings = []
        start_idx = 0

        # num_layer (we use default = 4), batch_size (concatenated visits), seq_len (clinical note sequences), hidden_dim.
        # average across num_layers and seq_len
        if strategy == 'mean':
            # batch_size (concatenated visits), hidden_dim.
            sentence_representation = torch.stack(encoder_outputs_all[-num_layers:]).mean(dim=[0, 2])

            for length in hospital_ids_lens:
                # We then average across visits
                # batch_size (true batch size), hidden_dim.
                batch_embeddings.append(torch.mean(sentence_representation[start_idx:start_idx + length],dim=0))
                start_idx += length
        
            return torch.stack(batch_embeddings)
    
        elif strategy == 'concat':
            # num_layer, batch_size (concatenated visits), hidden_dim.
            sentence_representation = torch.stack(encoder_outputs_all[-num_layers:]).mean(dim=2)

            for length in hospital_ids_lens:
                # We then average across visits
                # num_layer, batch_size (true batch size), hidden_dim.
                batch_embeddings.append(torch.mean(sentence_representation[:,start_idx:start_idx + length],dim=1))
                start_idx += length
            
            return torch.stack(batch_embeddings)
        
        elif strategy == 'all':
            # num_layer, batch_size (concatenated visits), seq_len (clinical note sequences), hidden_dim.
            sentence_representation = torch.stack(encoder_outputs_all[-num_layers:])
            return sentence_representation
        else:
            raise ValueError(f'{strategy} is not a valid strategy, choose between mean and concat')
        

        
        