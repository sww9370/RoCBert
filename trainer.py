import logging
import os

import numpy as np
import torch
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler
from tqdm import trange, tqdm
from transformers import AdamW, get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)


def get_char_accuracy(pre_labels, target_labels):
    num_all = len(target_labels)
    num_right = 0
    for i in range(len(target_labels)):
        is_right = True
        for j in range(len(target_labels[i])):
            if target_labels[i][j] == 102:
                break
            if target_labels[i][j] != pre_labels[i][j]:
                is_right = False
                break
        if is_right:
            num_right += 1
    return num_right / num_all


class Trainer(object):
    def __init__(self,
                 data_collator,
                 eval_data_collator=None,
                 gradient_accumulation_steps: int = 1,
                 weight_decay: float = 0.0, max_grad_norm: float = 1.0):
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.weight_decay = weight_decay
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = "cpu"
        self.max_grad_norm = max_grad_norm
        self.data_collator = data_collator
        self.eval_data_collator = data_collator if eval_data_collator is None else eval_data_collator
        self.num_gpu = torch.cuda.device_count()  # 支持多卡训练

    def evaluate(self, eval_dataset, eval_batch_size, model):
        sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(
            eval_dataset,
            sampler=sampler,
            batch_size=eval_batch_size,
            collate_fn=self.eval_data_collator
        )
        eval_losses = []
        model.eval()
        real_labels = None
        pre_labels = None
        for inputs in tqdm(eval_dataloader, desc="Evaluation"):
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(self.device)
            if real_labels is None:
                real_labels = inputs['labels_input_ids'][:, 0:]
            else:
                real_labels = torch.cat((real_labels, inputs['labels_input_ids'][:, 0:]), dim=0)
            with torch.no_grad():
                outputs = model(**inputs)

                step_eval_loss = outputs[0]
                if pre_labels is None:
                    pre_labels = torch.argmax(outputs[1], 2)
                else:
                    pre_labels = torch.cat((pre_labels, torch.argmax(outputs[1], 2)), dim=0)
                eval_losses += [step_eval_loss.mean().item()]
        eval_loss = np.mean(eval_losses)
        char_acc = get_char_accuracy(pre_labels, real_labels)
        return eval_loss, char_acc

    def train(self, train_datasets,  # dict_format; [{data: tensor, epoch:1}, {data:tensor, epoch:2}]
              model,
              checkpoint_dir: str,
              eval_dataset,
              eval_batch_size: int = 64,
              batch_size: int = 16,
              learning_rate: float = 5e-5, adam_epsilon: float = 1e-8,
              warmup_steps: float = 0, log_interval: int = 20, step_to_save: int = 0, restore_path=None):
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)

        dataloader_info = list()
        t_total = 0
        for cur_datset in train_datasets:
            train_sampler = RandomSampler(cur_datset["data"])
            data_collate_fn = cur_datset.get("collate_fn", self.data_collator)
            train_dataloader = DataLoader(cur_datset["data"], sampler=train_sampler, batch_size=batch_size,
                                          collate_fn=data_collate_fn)
            cur_epoch = cur_datset["epoch"]
            dataloader_info.append({"dataloader": train_dataloader, "epoch": cur_epoch})
            t_total += len(train_dataloader) // self.gradient_accumulation_steps * cur_epoch

        if restore_path is not None:
            logger.info(f"start restore from checkpoint {restore_path}")
            model.load_state_dict(torch.load(f"{restore_path}/pytorch_model.bin", map_location=self.device))
        num_gpu = torch.cuda.device_count()  # enable multi-gpu
        logger.info(f"num_gpu={num_gpu}")
        if num_gpu > 1:
            model = torch.nn.DataParallel(model)
        model.to(self.device)

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]
        # Prepare optimizer and schedule (linear warmup and decay)
        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=t_total)

        if restore_path is not None:
            optimizer.load_state_dict(torch.load(f"{restore_path}/optimizer.pt", map_location=self.device))
            scheduler.load_state_dict(torch.load(f"{restore_path}/scheduler.pt", map_location=self.device))
            logger.info(f"restore from checkpoint {restore_path} done")

        model.zero_grad()
        total_step = 0
        eval_loss_info = dict()
        char_acc_info = dict()
        max_acc = 0.0
        min_loss = 99999999
        for cur_info in dataloader_info:
            for eid in trange(0, cur_info["epoch"]):
                tr_loss, logging_loss = 0.0, 0.0
                for step, inputs in enumerate(tqdm(cur_info["dataloader"], desc='Iteration')):
                    model.train()
                    # counting loss
                    for k, v in inputs.items():
                        if isinstance(v, torch.Tensor):
                            inputs[k] = v.to(self.device)
                    outputs = model(**inputs)
                    loss = outputs[0]
                    if self.gradient_accumulation_steps > 1:
                        loss /= self.gradient_accumulation_steps
                    if self.num_gpu > 1:
                        loss = loss.mean()  # for data parall couting
                    loss.backward()
                    # tr_loss += loss.item()
                    tr_loss += float(loss)

                    if (step + 1) % self.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
                        optimizer.step()
                        scheduler.step()
                        model.zero_grad()

                    if step >= 6000 or restore_path is not None:
                        if self.num_gpu < 2:
                            if not model.roc_bert.embeddings.shape_embed.weight.requires_grad or not model.roc_bert.embeddings.pronunciation_embed.weight.requires_grad:
                                model.roc_bert.embeddings.shape_embed.weight.requires_grad = True
                                model.roc_bert.embeddings.pronunciation_embed.weight.requires_grad = True
                        else:
                            if not model.module.roc_bert.embeddings.shape_embed.weight.requires_grad or not model.module.roc_bert.embeddings.pronunciation_embed.weight.requires_grad:
                                model.module.roc_bert.embeddings.shape_embed.weight.requires_grad = True
                                model.module.roc_bert.embeddings.pronunciation_embed.weight.requires_grad = True
                    if step > 0 and step % log_interval == 0:
                        if self.num_gpu < 2:
                            requires_grad_shape = model.roc_bert.embeddings.shape_embed.weight.requires_grad
                            requires_grad_proun = model.roc_bert.embeddings.pronunciation_embed.weight.requires_grad
                        else:
                            requires_grad_shape = model.module.roc_bert.embeddings.shape_embed.weight.requires_grad
                            requires_grad_proun = model.module.roc_bert.embeddings.pronunciation_embed.weight.requires_grad
                        logger.warning(
                            f'step={step}, loss={tr_loss / step}, cur_loss={loss.item()}, requires_grad_shape={requires_grad_shape}, requires_grad_proun={requires_grad_proun}')
                    total_step += 1
                    if step_to_save != 0 and total_step % step_to_save == 0 and (
                            eid >= 0 or restore_path is not None):
                        # eval dataset every step_to_save and save the best model
                        if eval_dataset is not None:
                            eval_loss, char_acc = self.evaluate(eval_dataset, eval_batch_size, model)
                            eval_loss_info[total_step] = eval_loss
                            char_acc_info[total_step] = char_acc
                            min_loss = eval_loss if eval_loss < min_loss else min_loss
                            max_acc = char_acc if char_acc > max_acc else max_acc
                            logger.warning(f'step={total_step}, eval_loss={eval_loss}, char_acc={char_acc}')
                            logger.warning(f'history eval_loss_info: {eval_loss_info}')
                            logger.warning(f'history char_acc_info: {char_acc_info}')
                            if char_acc >= max_acc:  # update model if model acc is higher than previous
                                logger.warning(f"save the model at char_acc={char_acc}")
                                try:
                                    model_to_save = (model.module if hasattr(model,
                                                                             "module") else model)  # Take care of distributed/parallel training
                                    save_dir = f'{checkpoint_dir}/best_model_with_step'
                                    os.makedirs(save_dir, exist_ok=True)
                                    model_to_save.save_pretrained(save_dir)
                                    torch.save(optimizer.state_dict(), os.path.join(save_dir, "optimizer.pt"))
                                    torch.save(scheduler.state_dict(), os.path.join(save_dir, "scheduler.pt"))
                                except Exception as e:
                                    logger.info(f"failed to save model, err_msg: {e}")

                # evaluate at every epoch
                if eval_dataset is not None:
                    eval_loss, char_acc = self.evaluate(eval_dataset, eval_batch_size, model)
                    eval_loss_info[eid] = eval_loss
                    char_acc_info[eid] = char_acc
                    min_loss = eval_loss if eval_loss < min_loss else min_loss
                    max_acc = char_acc if char_acc > max_acc else max_acc
                    logger.warning(f'eid={eid}, eval_loss={eval_loss}, char_acc={char_acc}')
                    logger.warning(f'history eval_loss_info: {eval_loss_info}')
                    logger.warning(f'history char_acc_info: {char_acc_info}')
                try:
                    model_to_save = (model.module if hasattr(model,
                                                             "module") else model)  # Take care of distributed/parallel training
                    save_dir = f'{checkpoint_dir}/epoch_{eid}'
                    os.makedirs(save_dir, exist_ok=True)
                    model_to_save.save_pretrained(save_dir)
                    torch.save(optimizer.state_dict(), os.path.join(save_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(save_dir, "scheduler.pt"))
                except Exception as e:
                    logger.info(f"failed to save model, err_msg: {e}")
