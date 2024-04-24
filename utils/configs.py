from dataclasses import dataclass

@dataclass
class Config:
    seed : int = None
    strategy = 'SDP'
    predict_procedure : bool = False
    predict_drugs : bool = False
    procedure : bool = not(predict_procedure)
    drugs : bool = not(predict_drugs)
    truncate : bool = True
    pad : bool = True
    input_max_length :int = 448
    target_max_length :int = 64
    test_size : float = 0.05
    valid_size : float = 0.05
    source_vocab_size : int = None
    target_vocab_size : int = None
    num_encoder_layers: int = 5
    num_decoder_layers: int = 5
    nhead: int = 2
    emb_size: int = 512
    ffn_hid_dim: int = 2048
    train_batch_size: int = 64
    eval_batch_size: int = 256
    learning_rate: float = 3e-4
    warmup_start: float = 5
    num_train_epochs: int = 25
    warmup_epochs: int = None
    label_smoothing : float = 0.0