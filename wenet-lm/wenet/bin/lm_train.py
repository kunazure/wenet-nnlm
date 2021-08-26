# Copyright 2021 NWPU(Kun Wei)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


"""LM training in pytorch."""

import copy
import json
import logging
import os
import random
import subprocess
import sys

import configargparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.nn.parallel import data_parallel
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from wenet.transformer.lm_model import SequentialRNNLM
from wenet.utils.checkpoint import load_checkpoint, save_checkpoint
from wenet.utils.common import (IGNORE_ID, add_sos_eos, log_add,
                                remove_duplicates_and_blank, reverse_pad_list,
                                th_accuracy)
from wenet.utils.executor import LMExecutor
from wenet.utils.lm_utils import count_tokens, load_dataset, read_tokens
from wenet.utils.scheduler import WarmupLR

"""Language model training script."""

# NOTE: you need this func to generate our sphinx doc
def get_parser(parser=None, required=True):
    """Get parser."""
    if parser is None:
        parser = configargparse.ArgumentParser(
            description="Train a new language model on one CPU or one GPU",
            config_file_parser_class=configargparse.YAMLConfigFileParser,
            formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
        )
    # general configuration
    parser.add("--config", is_config_file=True, help="config file path")
    parser.add_argument(
        "--ngpu",
        default=None,
        type=int,
        help="Number of GPUs. If not given, use all visible devices",
    )
    parser.add_argument(
        "--outdir", type=str, required=required, help="Output directory"
    )
    parser.add_argument("--dict", type=str, required=required, help="Dictionary")
    parser.add_argument("--seed", default=1, type=int, help="Random seed")
    parser.add_argument(
        "--lm_checkpoint",
        "-r",
        default=None,
        nargs="?",
        help="Resume the training from snapshot",
    )
    # task related
    parser.add_argument(
        "--train-label",
        type=str,
        required=required,
        help="Filename of train label data",
    )
    parser.add_argument(
        "--valid-label",
        type=str,
        required=required,
        help="Filename of validation label data",
    )
    parser.add_argument("--test-label", type=str, help="Filename of test label data")
    parser.add_argument(
        "--dump-hdf5-path",
        type=str,
        default=None,
        help="Path to dump a preprocessed dataset as hdf5",
    )
    # training configuration
    parser.add_argument("--opt", default="sgd", type=str, help="Optimizer")
    parser.add_argument(
        "--sortagrad",
        default=0,
        type=int,
        nargs="?",
        help="How many epochs to use sortagrad for. 0 = deactivated, -1 = all epochs",
    )
    parser.add_argument(
        "--batchsize",
        "-b",
        type=int,
        default=300,
        help="Number of examples in each mini-batch",
    )
    parser.add_argument(
        "--accum-grad", type=int, default=1, help="Number of gradient accumueration"
    )
    parser.add_argument(
        "--epoch",
        "-e",
        type=int,
        default=20,
        help="Number of sweeps over the dataset to train",
    )
    parser.add_argument(
        "--early-stop-criterion",
        default="validation/main/loss",
        type=str,
        nargs="?",
        help="Value to monitor to trigger an early stopping of the training",
    )
    parser.add_argument(
        "--patience",
        default=3,
        type=int,
        nargs="?",
        help="Number of epochs "
        "to wait without improvement before stopping the training",
    )
    parser.add_argument(
        "--schedulers",
        default=None,
        action="append",
        type=lambda kv: kv.split("="),
        help="optimizer schedulers, you can configure params like:"
        " <optimizer-param>-<scheduler-name>-<schduler-param>"
        ' e.g., "--schedulers lr=noam --lr-noam-warmup 1000".',
    )
    parser.add_argument(
        "--gradclip",
        "-c",
        type=float,
        default=5,
        help="Gradient norm threshold to clip",
    )
    parser.add_argument(
        "--maxlen",
        type=int,
        default=40,
        help="Batch size is reduced if the input sequence > ML",
    )
    parser.add_argument(
        "--model-module",
        type=str,
        default="default",
        help="model defined module "
        "(default: espnet.nets.xxx_backend.lm.default:DefaultRNNLM)",
    )
    return parser

def compute_perplexity(result):
    """Compute and add the perplexity to the LogReport.

    :param dict result: The current observations
    """
    # Routine to rewrite the result dictionary of LogReport to add perplexity values
    result["perplexity"] = np.exp(result["main/nll"] / result["main/count"])
    if "validation/main/nll" in result:
        result["val_perplexity"] = np.exp(
            result["validation/main/nll"] / result["validation/main/count"]
        )


def concat_examples(batch, device=None, padding=None):
    """Concat examples in minibatch.

    :param np.ndarray batch: The batch to concatenate
    :param int device: The device to send to
    :param Tuple[int,int] padding: The padding to use
    :return: (inputs, targets)
    :rtype (torch.Tensor, torch.Tensor)
    """
    x, t = convert.concat_examples(batch, padding=padding)
    x = torch.from_numpy(x)
    t = torch.from_numpy(t)
    if device is not None and device >= 0:
        x = x.cuda(device)
        t = t.cuda(device)
    return x, t

class TextDataset(Dataset):
    def __init__(self, text_list):
        self.text_data = text_list

    def __len__(self):
        return len(self.text_data)

    def __getitem__(self, idx):
        return self.text_data[idx]

def collate_function(data):
    data_list = [x for x in data]
    return data_list

def main(cmd_args):
    """Train LM."""
    parser = get_parser()
    args, _ = parser.parse_known_args(cmd_args)
    # Save configs to model_dir/train.yaml for inference and export
    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    
    ngpu = args.ngpu
    logging.info(f"ngpu: {ngpu}")

    # seed setting
    nseed = args.seed
    random.seed(nseed)
    np.random.seed(nseed)

    # load dictionary
    with open(args.dict, "rb") as f:
        dictionary = f.readlines()
    char_list = [entry.decode("utf-8").split(" ")[0] for entry in dictionary]
    char_list.insert(0, "<blank>")
    char_list.append("<eos>")
    args.char_list_dict = {x: i for i, x in enumerate(char_list)}
    args.n_vocab = len(char_list)
    configs['n_vocab'] = args.n_vocab
    model = SequentialRNNLM(configs)
    print(model)

    """Train with the given args.
    """
    # check cuda and cudnn availability
    if not torch.cuda.is_available():
        logging.warning("cuda is not available")

    # get special label ids
    unk = args.char_list_dict["<unk>"]
    eos = args.char_list_dict["<eos>"]
    # read tokens as a sequence of sentences
    cv, n_cv_tokens, n_cv_oovs = load_dataset(
        args.valid_label, args.char_list_dict, args.dump_hdf5_path
    )
    train, n_train_tokens, n_train_oovs = load_dataset(
        args.train_label, args.char_list_dict, args.dump_hdf5_path
    )
    logging.info("#vocab = " + str(args.n_vocab))
    logging.info("#sentences in the training data = " + str(len(train)))
    logging.info("#tokens in the training data = " + str(n_train_tokens))
    logging.info(
        "oov rate in the training data = %.2f %%"
        % (n_train_oovs / n_train_tokens * 100)
    )

    #dataloader
    train_sampler = None
    cv_sampler = None
    train_dataset = TextDataset(train)
    cv_dataset = TextDataset(cv)
    train_data_loader = DataLoader(train_dataset,
                                   collate_fn=collate_function,
                                   sampler=train_sampler,
                                   shuffle=(train_sampler is None),
                                   batch_size=args.batchsize,)
    cv_data_loader = DataLoader(cv_dataset,
                                sampler=cv_sampler,
                                collate_fn=collate_function,
                                shuffle=False,
                                batch_size=args.batchsize,)

    

    use_sortagrad = args.sortagrad == -1 or args.sortagrad > 0
    # Create the dataset iterators
    batch_size = args.batchsize
    model_dir = args.outdir

    if args.lm_checkpoint is not None:
        infos = load_checkpoint(model, args.lm_checkpoint)
    else:
        infos = {}
    start_epoch = infos.get('epoch', -1) + 1
    cv_loss = infos.get('cv_loss', 0.0)
    step = infos.get('step', -1)
    num_epochs = configs.get('max_epoch', 100)

    # Save model conf to json
    saved_config_path = os.path.join(args.outdir, 'train.yaml')
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    with open(saved_config_path, 'w') as fout:
        data = yaml.dump(configs)
        fout.write(data)
    # Set up an optimizer
    executor = LMExecutor()

    optimizer = optim.Adam(model.parameters(), **configs['optim_conf'])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    use_cuda = args.ngpu >= 0 and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    model = model.to(device)
    if start_epoch == 0:
        save_model_path = os.path.join(model_dir, 'init.pt')
        save_checkpoint(model, save_model_path)
    
    # Start training loop
    executor.step = step
    # scheduler.set_step(step)
    scaler = None

    for epoch in range(start_epoch, num_epochs):
        lr = optimizer.param_groups[0]['lr']
        logging.info('Epoch {} TRAIN info lr {}'.format(epoch, lr))
        executor.train(model, optimizer, scheduler, train_data_loader, device,
                       configs)
        total_loss, num_seen_utts = executor.cv(model, cv_data_loader, device,
                                                configs)
        cv_loss = total_loss / num_seen_utts

        logging.info('Epoch {} CV info cv_loss {}'.format(epoch, cv_loss))
        save_model_path = os.path.join(model_dir, '{}.pt'.format(epoch))
        save_checkpoint(
            model, save_model_path, {
                'epoch': epoch,
                'lr': lr,
                'cv_loss': cv_loss,
                'step': executor.step
            })
        final_epoch = epoch

    if final_epoch is not None and args.rank == 0:
        final_model_path = os.path.join(model_dir, 'final.pt')
        os.symlink('{}.pt'.format(final_epoch), final_model_path)

    # compute perplexity for test set
    if args.test_label:
        logging.info("test the best model")
        torch_load(args.outdir + "/rnnlm.model.best", model)
        test = read_tokens(args.test_label, args.char_list_dict)
        n_test_tokens, n_test_oovs = count_tokens(test, unk)
        logging.info("#sentences in the test data = " + str(len(test)))
        logging.info("#tokens in the test data = " + str(n_test_tokens))
        logging.info(
            "oov rate in the test data = %.2f %%" % (n_test_oovs / n_test_tokens * 100)
        )
        # TODO
        # test_iter = ParallelSentenceIterator(
        #     test, batch_size, max_length=args.maxlen, sos=eos, eos=eos, repeat=False
        # )
        evaluator = LMEvaluator(test_iter, model, reporter, device=gpu_id)
        result = evaluator()
        compute_perplexity(result)
        logging.info(f"test perplexity: {result['perplexity']}")        

if __name__ == "__main__":
    main(sys.argv[1:])
