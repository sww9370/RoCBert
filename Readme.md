#RoCBert: Robust Chinese Bert with Multimodal Contrastive Pretraining

---

**Authors**: Hui Su, Weiwei Shi, Xiaoyu Shen, Xiao Zhou, Tuo Ji, Jiarui Fang, and Jie Zhou

This repository contains pretrain code for [RoCBert](https://aclanthology.org/2022.acl-long.65.pdf), model is avaliable at [huggingface model hub](https://huggingface.co/weiweishi/roc-bert-base-zh). 

If you find our paper and resources useful, please kindly leave a star and cite our papers. Thanks!

```bibtex
@inproceedings{su2022rocbert,
title={RoCBert: Robust Chinese Bert with Multimodal Contrastive Pretraining},
author={Su, Hui and Shi, Weiwei and Shen, Xiaoyu and Xiao, Zhou and Ji, Tuo and Fang, Jiarui and Zhou, Jie},
booktitle={Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
pages={921--931},
year={2022}
}
```


## News
---
* [2022/12/01] RoCBert is publicly released!



## Requirements

---

* Python>=3.8
* transformers==4.20.1
* torch==1.12.0
* pypinyin


## QuickTour

---

**Pretrain**

```shell
python3 execute_pretrain.py \
--train_data_dir="train_data" `#path of train data` \
--dev_data_dir="dev_data" `#path of dev data` \
--output_dir="dev_data" `#path to save the checkpoint` \
--tokenizer_path="dev_data" `#path of tokenizer` \
--restore_path `#path of checkpoint for restore"`\
--merge_input_way="attention" `#defines the way of merging the shape_embed, pronunciation_embed and word_embed, includs attention, concat, plus` \
--enable_cls=True `#whether to enable contrastive learning based loss` \
--max_seq_length=128 `` \
--sen_attack_ratio=0.1 `#sentence attack ratio while pretrain` \
--attack_ratio=0.15 `#character attack ratio for each sentence should attack while pretrain` \
--sen_mask_ratio=0.1 `#sentence mask ratio while pretrain` \
--mask_ratio=0.15 `#character mask ratio for each sentence should mask while pretrain` \
--learning_rate=1e-4 `#the initial learning rate for Adam` \
--warmup_steps=10000 `#warm up steps of training"` \
--train_batch_size=384 `#batch size for training` \
--eval_batch_size=384 `#batch size for evaluation` \
--step_to_save=5000 `#save for every * steps` \
--num_train_epochs=500 `#total number of training epochs to perform` \
--weight_decay=0.01 `#weight deay of training` \
--gradient_accumulation_steps=7 `#gradient_accumulation_steps` \
--enable_pronunciation=True `#whether or not the model use pronunciation embed when training` \
--pronunciation_embed_dim=768 `#dimension of the pronunciation_embed` \
--pronunciation_vocab_size=910 `#pronunciation vocabulary size of the RoCBert model` \
--word_pronunciation_path="train_data/word_pronunciation.json" `#word<->pronunciation map dict, such as {word: proun_id}` \
--pronunciation_id_path="train_data/prounciation.proun_id" `#proud<->id map list` \
--enable_shape=True `#whether or not the model use shape embed when training` \
--shape_embed_dim=512 `#dimension of the shape_embed` \
--shape_vocab_size=24858 `#shape vocabulary size of the RoCBert model` \
--word_shape_path="train_data/word_shape.json" `#word<->shape map dict, such as {word: shape_id}`\
--shape_embed_path="train_data/img_embed.shape_embed" `#shape<->embed map list`

```
