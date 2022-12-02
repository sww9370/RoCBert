from dataset import *
from trainer import Trainer
from modeling_roc_bert import RoCBertForPreTraining
from tokenization_roc_bert import RoCBertTokenizer
from configuration_roc_bert import RoCBertConfig
import logging
import torch
from torch import nn
import argparse
import os
import numpy as np
import random
from attacker.building import DataBuilder
logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(message)s')
logger = logging.getLogger(__name__)



def train_advance(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"start load dataset")
    tokenizer = RoCBertTokenizer.from_pretrained(args.tokenizer_path)
    attacker = DataBuilder()
    word_shape, shape_embed = LargePretrainDataOnlineAttack.load_shape_info(args.word_shape_path, args.shape_embed_path)
    word_pronunciation, pronunciation_id, pronunciation_embed = LargePretrainDataOnlineAttack.load_pronunciation_info(args.word_pronunciation_path, args.pronunciation_id_path, args.pronunciation_embed_path)
    if shape_embed is not None:
        shape_embed = torch.tensor(shape_embed, device=device)
    if pronunciation_embed is not None:
        pronunciation_embed = torch.tensor(pronunciation_embed, device=device)

    train_filenames = os.listdir(args.train_data_dir)
    for i in range(len(train_filenames)):
        train_filenames[i] = f"{args.train_data_dir}/{train_filenames[i]}"
    dev_filenames = os.listdir(args.dev_data_dir)
    for i in range(len(dev_filenames)):
        dev_filenames[i] = f"{args.dev_data_dir}/{dev_filenames[i]}"
    logger.info(f"train_filenames={train_filenames}, dev_filenames={dev_filenames}")

    train_dataset = LargePretrainDataOnlineAttack(train_filenames, word_pronunciation, word_shape, tokenizer, args.max_seq_length,
                                                  sen_mask_ratio=args.sen_mask_ratio, mask_ratio=args.mask_ratio, sen_attack_ratio=args.sen_attack_ratio, attack_ratio=args.attack_ratio,
                                                  attacker=attacker, bpe_tokenizer=None, bpe_share_py=args.bpe_share_py, proun_vocab_size=args.pronunciation_vocab_size,
                                                  shape_vocab_size=args.shape_vocab_size, mask_all=True,
                                                  )
    dev_dataset = LargePretrainDataOnlineAttack(dev_filenames, word_pronunciation, word_shape, tokenizer, args.max_seq_length, attacker=attacker)
    logger.info(f"loaded dataset")

    logger.info("start init model")
    roc_bert_config = RoCBertConfig()
    roc_bert_config.pronunciation_embed_dim = args.pronunciation_embed_dim
    roc_bert_config.shape_embed_dim = args.shape_embed_dim
    roc_bert_config.pronunciation_vocab_size = args.pronunciation_vocab_size
    roc_bert_config.shape_vocab_size = args.shape_vocab_size
    roc_bert_config.enable_cls = args.enable_cls
    roc_bert_config.enable_shape = args.enable_shape
    roc_bert_config.enable_pronunciation = args.enable_pronunciation
    roc_bert_config.vocab_size = tokenizer.vocab_size
    roc_bert_config.merge_input_way = args.merge_input_way
    roc_bert_config.additional_attention_layer = 1

    model = RoCBertForPreTraining(roc_bert_config)
    if pronunciation_embed is None:
        pronunciation_embed = torch.randn(len(pronunciation_id), roc_bert_config.hidden_size)
    model.roc_bert.embeddings.pronunciation_embed.weight = nn.Parameter(pronunciation_embed)
    model.roc_bert.embeddings.shape_embed.weight = nn.Parameter(shape_embed)
    model.roc_bert.embeddings.shape_embed.weight.requires_grad = False
    model.to(device)
    logger.info(f"inited model")

    logger.info("start train")
    trainer = Trainer(LargePretrainDataOnlineAttack.collate_dict, weight_decay=args.weight_decay, gradient_accumulation_steps=args.gradient_accumulation_steps)
    trainer.train(model=model,
                  train_datasets=[{"data": train_dataset, "epoch": args.num_train_epochs}],
                  eval_dataset=dev_dataset,
                  learning_rate=args.learning_rate,
                  warmup_steps=args.warmup_steps,
                  batch_size=args.train_batch_size,
                  eval_batch_size=args.eval_batch_size,
                  checkpoint_dir=args.output_dir,
                  step_to_save=args.step_to_save,
                  restore_path=args.restore_path,
                  )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--merge_input_way", default="attention", type=str, required=False, help="Defines the way of merging the shape_embed, pronunciation_embed and word_embed, includs attention, concat, plus")
    parser.add_argument("--enable_cls", default=True, type=bool, required=False)
    parser.add_argument("--enable_pronunciation", default=True, type=bool, required=False)
    parser.add_argument("--pronunciation_embed_dim", default=768, type=int, required=False)
    parser.add_argument("--pronunciation_vocab_size", default=910, type=int, required=False)
    parser.add_argument("--word_pronunciation_path", default="train_data/word_pronunciation.json", type=str, required=False,
                        help="word<->pronunciation map dict, such as {word: proun_id}")
    parser.add_argument("--pronunciation_id_path", default="train_data/prounciation.proun_id", type=str, required=False,
                        help="proud<->id map list")
    parser.add_argument("--pronunciation_embed_path", default=None, type=str, required=False,
                        help="proud<->id map list")

    parser.add_argument("--enable_shape", default=True, type=bool, required=False)
    parser.add_argument("--shape_embed_dim", default=512, type=int, required=False)
    parser.add_argument("--shape_vocab_size", default=24858, type=int, required=False)
    parser.add_argument("--word_shape_path", default="train_data/word_shape.json", type=str, required=False,
                        help="word<->shape map dict, such as {word: shape_id}")
    parser.add_argument("--shape_embed_path", default="train_data/img_embed.shape_embed", type=str, required=False,
                        help="shape<->embed map list")

    parser.add_argument("--max_seq_length", default=128, type=int, required=False, help="the maximum total input sequence length after tokenization")
    parser.add_argument("--sen_mask_ratio", default=1.0, type=float, help="sentence mask ratio while pretrain", required=False)
    parser.add_argument("--mask_ratio", default=0.15, type=float, help="character mask ratio for each sentence should mask while pretrain", required=False)
    parser.add_argument("--sen_attack_ratio", default=0.1, type=float, help="sentence attack ratio while pretrain", required=False)
    parser.add_argument("--attack_ratio", default=0.15, type=float, help="character attack ratio for each sentence should attack while pretrain", required=False)

    parser.add_argument("--learning_rate", default=1e-4, type=float, help="the initial learning rate for Adam", required=False)
    parser.add_argument("--warmup_steps", default=10000, type=int, help="warm up steps of training", required=False)
    parser.add_argument("--train_batch_size", default=384, type=int, help="batch size for training", required=False)
    parser.add_argument("--eval_batch_size", default=384, type=int, help="batch size for evaluation", required=False)
    parser.add_argument("--step_to_save", default=5000, type=int, help="save for every * steps", required=False)
    parser.add_argument("--restore_path", default=None, type=str, help="path of checkpoint for restore", required=False)
    parser.add_argument("--num_train_epochs", default=500, type=int, help="total number of training epochs to perform.", required=False)
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--gradient_accumulation_steps", default=7, type=int, help="gradient_accumulation_steps")
    parser.add_argument("--output_dir", default="models", type=str, required=False, help="the output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--tokenizer_path", default="train_data/", type=str, required=False)
    parser.add_argument("--bpe_tokenizer_path", default=None, type=str, required=False)
    parser.add_argument("--bpe_vocab_path", default=None, type=str, required=False)
    parser.add_argument("--bpe_share_py", default=True, type=bool, required=False)

    parser.add_argument("--train_data_dir", default="", type=str, required=False)
    parser.add_argument("--dev_data_dir", default="", type=str, required=False)

    parser.add_argument('--seed', type=int, default=666, help="random seed for initialization")
    args = parser.parse_args()

    seed = args.seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    train_advance(args)


if __name__ == "__main__":
    main()
