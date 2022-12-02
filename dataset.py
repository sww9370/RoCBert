from torch.utils.data import Dataset
import logging
import random, json
import torch
logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(message)s')
logger = logging.getLogger(__name__)

class LargePretrainDataOnlineAttack(Dataset):
    def __init__(self, filenames, word_prounciation, word_shape, tokenizer, max_len, sen_mask_ratio=0.0,
                 mask_ratio=0.0, sen_attack_ratio=0.0, attack_ratio=0.0, data_length=None, attacker=None, bpe_tokenizer=None,
                 bpe_share_py=False, proun_vocab_size=None, shape_vocab_size=None, mask_all=False):
        self.filenames = filenames
        random.shuffle(self.filenames)
        self.index_filename = 0
        self.index_data = 0
        self.querys = list()
        self.query_important_words = list()
        self.data_length = data_length
        if self.data_length == None:
            cur_len = 0
            for cur_filename in self.filenames:
                data_info = LargePretrainDataOnlineAttack.load_data([cur_filename], shuffle=False)
                cur_len += len(data_info["querys"])
            self.data_length = cur_len
            logger.info(f"there are total {self.data_length} train data")

        self.word_prounciation = word_prounciation
        self.word_shape = word_shape
        self.tokenizer = tokenizer
        self.bpe_tokenizer = bpe_tokenizer
        self.bpe_share_py = bpe_share_py
        self.vocab_size = len(self.tokenizer.vocab)
        self.proun_vocab_size = proun_vocab_size
        self.shape_vocab_size = shape_vocab_size

        self.word_unk_id = self.tokenizer.convert_tokens_to_ids("[UNK]")
        self.word_mask_id = self.tokenizer.convert_tokens_to_ids("[MASK]")
        logger.info(f"self.word_mask_id={self.word_mask_id}, self.word_unk_id={self.word_unk_id}")
        self.shape_unk_id = self.word_shape["[UNK]"]
        self.shape_mask_id = self.word_shape["[MASK]"]
        self.proun_unk_id = self.word_prounciation["[UNK]"]
        self.proun_mask_id = self.word_prounciation["[MASK]"]
        self.max_len = max_len
        self.mask_ratio = mask_ratio
        self.attack_ratio = attack_ratio
        self.sen_mask_ratio = sen_mask_ratio
        self.sen_attack_ratio = sen_attack_ratio
        self.mask_all = mask_all   # 在执行mask操作的时候，是否对全部的向量（即字音和字形向量）mask

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attacker = attacker

    def __len__(self):
        return self.data_length

    def get_bpe_token_res(self, cur_text):
        tokens = list()
        tokens_to_bpe = list()
        for i in range(len(cur_text)):
            cur_char = cur_text[i]
            if u'\u4e00' <= cur_char <= u'\u9fa5': # or len(self.bpe_tokenizer.encode(cur_char).tokens) == 0:  # 中文字符 或 非中文且bpe没结果
                if len(tokens_to_bpe) > 0:   # 先对非中文字符进行切割
                    in_str = ("".join(tokens_to_bpe)).strip()
                    bpe_res = self.bpe_tokenizer.encode(in_str).tokens
                    for cur_token in bpe_res:
                        tokens.append(cur_token)
                    tokens_to_bpe = list()
                tokens.append(cur_char)    # 如果该字符为中文，则直接append
            else:
                tokens_to_bpe.append(cur_char)
        if len(tokens_to_bpe) > 0:
            in_str = "".join(tokens_to_bpe)
            bpe_res = self.bpe_tokenizer.encode(in_str).tokens
            for cur_token in bpe_res:
                tokens.append(cur_token)
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        return tokens, tokens

    def get_bpe_token_res_share_pinyin(self, cur_text):
        # 输入的word id不变，然后共享拼音，例如 'tianqi" 的 word: "t i a n q i", pinyin: "tian tian tian tian qi qi"
        tokens = list()
        pinyin_tokens = list()
        tokens_to_bpe = list()
        for i in range(len(cur_text)):
            cur_char = cur_text[i]
            if u'\u4e00' <= cur_char <= u'\u9fa5': # or len(self.bpe_tokenizer.encode(cur_char).tokens) == 0:  # 中文字符 或 非中文且bpe没结果
                if len(tokens_to_bpe) > 0:   # 先对非中文字符进行切割
                    in_str = ("".join(tokens_to_bpe)).strip()
                    bpe_res = self.bpe_tokenizer.encode(in_str).tokens
                    for cur_token in bpe_res:
                        for cur_token_char in cur_token:
                            tokens.append(cur_token_char)
                            pinyin_tokens.append(cur_token)
                    tokens_to_bpe = list()
                tokens.append(cur_char)    # 如果该字符为中文，则直接append
                pinyin_tokens.append(cur_char)
            else:
                tokens_to_bpe.append(cur_char)
        if len(tokens_to_bpe) > 0:
            in_str = "".join(tokens_to_bpe)
            bpe_res = self.bpe_tokenizer.encode(in_str).tokens
            for cur_token in bpe_res:
                for cur_token_char  in cur_token:
                    tokens.append(cur_token_char)
                    pinyin_tokens.append(cur_token)
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        pinyin_tokens = ["[CLS]"] +pinyin_tokens + ["[SEP]"]
        tokens_tmp = " ".join(tokens)
        pinyin_tokens_tmp = " ".join(pinyin_tokens)
        return tokens, pinyin_tokens


    def __getitem__(self, cur_id):
        if self.index_data >= len(self.querys):
            self.index_filename += 1
            self.index_data = 0
            self.querys = list()
            self.query_important_words = list()
        if self.index_filename >= len(self.filenames):
            self.index_filename = 0
            random.shuffle(self.filenames)
        if len(self.querys) == 0:
            cur_filename = self.filenames[self.index_filename]
            logger.info(f"start load data from file:{cur_filename}")
            # 读取数据
            data_info = LargePretrainDataOnlineAttack.load_data([cur_filename], shuffle=True)
            self.querys = data_info["querys"][:]
            if "query_important_words" in data_info:
                self.query_important_words = data_info["query_important_words"][:]
            else:
                self.query_important_words = list()
        idx = self.index_data
        self.index_data += 1

        cur_input_token = self.querys[idx]
        cur_label_token = self.querys[idx]
        attack_sample_token, _ = self.generate_attack_sample(self.querys[idx], self.query_important_words[idx] if len(self.query_important_words) > 0 else None , self.attack_ratio)
        if random.random() < self.sen_attack_ratio:
            cur_input_token, cur_label_token = self.generate_attack_sample(self.querys[idx], self.query_important_words[idx] if len(self.query_important_words) > 0 else None, self.attack_ratio)
            if len(cur_input_token) != len(cur_label_token):
                min_len = min(len(cur_input_token), len(cur_label_token))
                cur_input_token = cur_input_token[:min_len]
                cur_label_token = cur_label_token[:min_len]

        if self.bpe_tokenizer is None:
            cur_input_token = ["[CLS]"] + [token for token in cur_input_token] + ["[SEP]"]
            cur_label_token = ["[CLS]"] + [token for token in cur_label_token] + ["[SEP]"]
            attack_sample_token = ["[CLS]"] + [token for token in attack_sample_token] + ["[SEP]"]
            cur_input_py_token = cur_input_token
            cur_label_py_token = cur_label_token
            attack_sample_py_token = attack_sample_token
        else:
            if self.bpe_share_py:
                cur_input_token, cur_input_py_token = self.get_bpe_token_res_share_pinyin(cur_input_token)
                cur_label_token, cur_label_py_token = self.get_bpe_token_res_share_pinyin(cur_label_token)
                attack_sample_token, attack_sample_py_token = self.get_bpe_token_res_share_pinyin(attack_sample_token)
            else:
                cur_input_token, cur_input_py_token = self.get_bpe_token_res(cur_input_token)
                cur_label_token, cur_label_py_token = self.get_bpe_token_res(cur_label_token)
                attack_sample_token, attack_sample_py_token = self.get_bpe_token_res(attack_sample_token)

        cur_input_ids = [self.tokenizer.convert_tokens_to_ids(token) for token in cur_input_token]   # 今 天 天 qi 不 错
        cur_input_shape = [self.word_shape.get(word, self.shape_unk_id) for word in cur_input_token]
        cur_input_pronunciation = [self.word_prounciation.get(word, self.proun_unk_id) for word in cur_input_py_token]

        cur_label_ids = [self.tokenizer.convert_tokens_to_ids(token) for token in cur_label_token]
        cur_label_shape = [self.word_shape.get(word, self.shape_unk_id) for word in cur_label_token]
        cur_label_pronunciation = [self.word_prounciation.get(word, self.proun_unk_id) for word in cur_label_py_token]

        attack_sample_ids = [self.tokenizer.convert_tokens_to_ids(token) for token in attack_sample_token]
        attack_sample_shape = [self.word_shape.get(word, self.shape_unk_id) for word in attack_sample_token]
        attack_sample_pronunciation = [self.word_prounciation.get(word, self.proun_unk_id) for word in attack_sample_py_token]

        if random.random() < self.sen_mask_ratio:
            cur_input_ids, cur_input_pronunciation, cur_input_shape = self.generate_mask_sample(cur_input_ids, cur_input_pronunciation, cur_input_shape, cur_label_ids, mask_ratio=self.mask_ratio, mask_wrong=False)

        cur_input_ids = torch.tensor(cur_input_ids)
        cur_label_ids = torch.tensor(cur_label_ids)
        attack_sample_ids = torch.tensor(attack_sample_ids)
        cur_input_shape = torch.tensor(cur_input_shape)
        cur_label_shape = torch.tensor(cur_label_shape)
        attack_sample_shape = torch.tensor(attack_sample_shape)
        cur_input_pronunciation = torch.tensor(cur_input_pronunciation)
        cur_label_pronunciation = torch.tensor(cur_label_pronunciation)
        attack_sample_pronunciation = torch.tensor(attack_sample_pronunciation)

        if len(cur_input_ids) >= self.max_len:
            cur_input_ids = cur_input_ids[:self.max_len]
            cur_input_shape = cur_input_shape[:self.max_len]
            cur_input_pronunciation = cur_input_pronunciation[:self.max_len]
            attention_mask = torch.tensor([1] * self.max_len, dtype=torch.long)
        else:
            pad_len = self.max_len - len(cur_input_ids)
            attention_mask = torch.tensor([1] * len(cur_input_ids) + [0] * pad_len, dtype=torch.long)
            query_padding = torch.tensor([0] * pad_len, dtype=torch.long)
            cur_input_ids = torch.cat((cur_input_ids, query_padding), -1)
            cur_input_pronunciation = torch.cat((cur_input_pronunciation, query_padding), -1)
            cur_input_shape = torch.cat((cur_input_shape, query_padding), -1)

        if len(attack_sample_ids) >= self.max_len:
            attack_sample_ids = attack_sample_ids[:self.max_len]
            attack_sample_shape = attack_sample_shape[:self.max_len]
            attack_sample_pronunciation = attack_sample_pronunciation[:self.max_len]
            attack_sample_mask = torch.tensor([1] * self.max_len, dtype=torch.long)
        else:
            pad_len = self.max_len - len(attack_sample_ids)
            attack_sample_mask = torch.tensor([1] * len(attack_sample_ids) + [0] * pad_len, dtype=torch.long)
            sample_padding = torch.tensor([0] * pad_len, dtype=torch.long)
            attack_sample_ids = torch.cat((attack_sample_ids, sample_padding), -1)
            attack_sample_shape = torch.cat((attack_sample_shape, sample_padding), -1)
            attack_sample_pronunciation = torch.cat((attack_sample_pronunciation, sample_padding), -1)

        if len(cur_label_ids) >= self.max_len:
            cur_label_ids = cur_label_ids[:self.max_len]
            cur_label_pronunciation = cur_label_pronunciation[:self.max_len]
            cur_label_shape = cur_label_shape[:self.max_len]
        else:
            pad_len_label = self.max_len - len(cur_label_ids)
            label_padding = torch.tensor([-100] * pad_len_label, dtype=torch.long)
            cur_label_ids = torch.cat((cur_label_ids, label_padding), -1)

            zero_padding = torch.tensor([0] * pad_len_label, dtype=torch.long)
            cur_label_pronunciation = torch.cat((cur_label_pronunciation, zero_padding), -1)
            cur_label_shape = torch.cat((cur_label_shape, zero_padding), -1)

        return_info = {
            "input_ids": cur_input_ids.to(self.device),
            "input_shape_ids": cur_input_shape.to(self.device),
            "input_pronunciation_ids": cur_input_pronunciation.to(self.device),
            "attack_input_ids": attack_sample_ids.to(self.device),
            "attack_input_shape_ids": attack_sample_shape,
            "attack_input_pronunciation_ids": attack_sample_pronunciation,
            "attack_attention_mask": attack_sample_mask.to(self.device),
            "attention_mask": attention_mask.to(self.device),

            "labels_input_ids": cur_label_ids.to(self.device),
            "labels_input_shape_ids": cur_label_shape.to(self.device),
            "labels_input_pronunciation_ids": cur_label_pronunciation.to(self.device),
        }
        return return_info

    def get_diff_index(self, cur_input, cur_label):
        # 获取输入和输出有diff的index，用于后续生成负例和mask替换等。
        diff_index = list()
        for i in range(len(cur_input)):
            if cur_input[i] != cur_label[i]:
                diff_index.append(i)
        return diff_index

    def generate_mask_sample(self, cur_input_ids, cur_proun_ids, cur_shape_ids, cur_label_ids, mask_wrong=True, mask_ratio=0.03):
        # 构造mask样例，mask掉错误的位置/随机mask，mask的时候只mask word_embed, 其他不变
        new_input = cur_input_ids[:]
        new_proun = cur_proun_ids[:]
        new_shape = cur_shape_ids[:]
        if mask_wrong:
            # 对输入错误的地方进行mask
            diff_index = self.get_diff_index(cur_input_ids, cur_label_ids)
            for cur_index in diff_index:
                new_input[cur_index] = self.word_mask_id
                if self.mask_all:
                    new_proun[cur_index] = self.proun_mask_id
                    new_shape[cur_index] = self.shape_mask_id
        else:
            # 训练时sen_mask_ratio=1, mask_ratio=0.15，每个单词有0.15的概率来决定是否mask，若要mask，80%替换[MASK], 10%随机字，%10维持原样
            mask_index = list()
            for i in range(1, len(cur_input_ids)-1):    # 头尾特殊字段不mask
                if random.random() < mask_ratio:
                    mask_index.append(i)
            for cur_index in mask_index:
                if random.random() < 0.1:
                    pass
                elif random.random() < 0.2:
                    new_input[cur_index] = random.randrange(self.vocab_size)
                    if self.mask_all:
                        new_proun[cur_index] = random.randrange(self.proun_vocab_size)
                        new_shape[cur_index] = random.randrange(self.shape_vocab_size)
                else:
                    new_input[cur_index] = self.word_mask_id
                    if self.mask_all:
                        new_proun[cur_index] = self.proun_mask_id
                        new_shape[cur_index] = self.shape_mask_id
        return new_input, new_proun, new_shape

    def generate_attack_sample(self, cur_query, cur_query_important_words, attack_ratio, random_attack=True):
        if len(cur_query) < 6:
            return cur_query, cur_query   # 不做变换

        attack_numbers = max(int(len(cur_query) * attack_ratio), 2)
        if random_attack or len(cur_query_important_words) is None:
            out_list = [i for i in range(len(cur_query))]
            indexs = random.sample(out_list, attack_numbers)
        else:
            out_list = cur_query_important_words[:attack_numbers]
            indexs = [li[1] - 1 for li in out_list]
        indexs.sort()
        start_index = 0
        noise_out = ""
        input_sen = ""
        for index in indexs:
            out = self.attacker.attack(cur_query[start_index:index + 1], index - start_index)
            blank_numbers = len(out) - (index + 1 - start_index)
            if blank_numbers < 0:
                noise_out += cur_query[start_index:index + 1]
                input_sen += cur_query[start_index:index + 1]
            else:
                noise_out += out
                input_sen += cur_query[start_index:index + 1] + "*" * blank_numbers
            start_index = index + 1
        noise_out += cur_query[start_index:]   # 攻击后的样本
        input_sen += cur_query[start_index:]   # 原始样本（待*符号的）
        return noise_out, input_sen

    @classmethod
    def collate_dict(cls, batch):
        new_batch = dict()
        if isinstance(batch[0], dict):
            exist_keys = set(batch[0].keys())
            for cur_key in exist_keys:
                new_batch[cur_key] = torch.squeeze(torch.stack([f[cur_key] for f in batch]))
            return new_batch

    @classmethod
    def load_shape_info(cls, word_shape_path, shape_embed_path):
        word_shape = json.load(open(word_shape_path, "r", encoding="utf8"))
        shape_embed = json.load(open(shape_embed_path, "r", encoding="utf8"))
        return word_shape, shape_embed

    @classmethod
    def load_pronunciation_info(cls, word_proun_path, proun_id_path, proun_embed_path):
        word_proun = json.load(open(word_proun_path, "r", encoding="utf8"))
        proun_id = json.load(open(proun_id_path, "r", encoding="utf8"))
        proun_embed = json.load(open(proun_embed_path, "r", encoding="utf8")) if proun_embed_path is not None else None
        return word_proun, proun_id, proun_embed

    @classmethod
    def load_data(cls, filenames, shuffle=True):
        ori_query = list()
        for filename in filenames:
            f_in = open(filename, "r", encoding="utf8")
            for line in f_in:
                ori_query.append(line.strip())
            f_in.close()
        logger.info(f"there are total {len(ori_query)} datas")
        if shuffle:
            random.shuffle(ori_query)
        return_info = {
            "querys": ori_query,
        }
        return return_info
