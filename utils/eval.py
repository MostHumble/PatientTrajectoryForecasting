import torch
from tqdm import tqdm
from utils.train import create_source_mask, generate_square_subsequent_mask

def evaluate(model, val_dataloader,  source_pad_id = 0, DEVICE='cuda:0', tgt_tokens_to_ids = None, max_len = 100):
    """
    Evaluate the model on the validation dataset.

    Args:
        model (torch.nn.Module): The model to be evaluated.
        val_dataloader (torch.utils.data.DataLoader): The validation dataloader.
        source_pad_id (int, optional): The padding token ID for the source input. Defaults to 0.
        DEVICE (str, optional): The device to run the evaluation on. Defaults to 'cuda:0'.
        tgt_tokens_to_ids (dict, optional): A dictionary mapping target tokens to their IDs. Defaults to None.
        max_len (int, optional): The maximum length of the generated target sequence. Defaults to 100.
    """

    model.eval()
    pred_trgs = []
    targets = []

    with torch.inference_mode():
        for source_input_ids, target_input_ids in tqdm(val_dataloader, desc='scoring'):
            print(source_input_ids.shape)
            source_input_ids, target_input_ids = source_input_ids.to(DEVICE),target_input_ids.to(DEVICE)
            src_mask, source_padding_mask = create_source_mask(source_input_ids, source_pad_id, DEVICE) 
            memory = model.batch_encode(source_input_ids, src_mask, source_padding_mask)
            pred_trg = torch.tensor(tgt_tokens_to_ids['BOS'], device= DEVICE).repeat(source_input_ids.size(0)).unsqueeze(1)
            # generate target sequence one token at a time at batch level
            for i in range(max_len):
                trg_mask = generate_square_subsequent_mask(i+1, DEVICE)
                output = model.decode(pred_trg, memory, trg_mask)
                probs = model.generator(output[:, -1])
                pred_tokens = torch.argmax(probs, dim=1)
                eov_mask = pred_tokens == tgt_tokens_to_ids['EOV']
                if eov_mask.any():
                    # extend with sequences that have reached EOV
                    pred_trgs.extend(torch.cat((pred_trg[eov_mask].cpu(),torch.tensor(tgt_tokens_to_ids['EOV']).unsqueeze(0).repeat(eov_mask.sum(), 1)),dim = -1).tolist())
                    targets.extend(target_input_ids[eov_mask].cpu().tolist())
                    # store corresponding target sequences
                    target_input_ids = target_input_ids[~eov_mask]
                    # break if all have reached EOV
                    if eov_mask.all():
                        break  
                    pred_trg = torch.cat((pred_trg[~eov_mask], pred_tokens[~eov_mask].unsqueeze(1)), dim=1)
                    memory = memory[~eov_mask]
                else:
                    pred_trg = torch.cat((pred_trg, pred_tokens.unsqueeze(1)), dim=1)
                
    # todo: test this function, implement mapk, and recall@k
