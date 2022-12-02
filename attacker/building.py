from .transform import *
from .homologous import *
import logging

#import jieba
# from find_important_words import *
# from transformers import BertTokenizer, BertConfig, BertForNextSentencePrediction, GPT2LMHeadModel
# import torch
# import re
# import copy


logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(message)s')
logger = logging.getLogger(__name__)



class DataBuilder():
    def __init__(self):
  #  self.tokenizer = tokenizer

        self.init_transforms()
        self.generated = []

    def init_transforms(self):
        random_replace_transform = RandomReplaceTransform('attacker/data/chaizi/hanzi.txt')
        token_swap_transform = TokenSwapTransform()
        char_swap_transform = CharSwapTransform()
        add_transform = AddTransform('attacker/data/chaizi/hanzi.txt')
        add_sep_transform = AddTransform('attacker/data/chaizi/hanzi.txt', '_ |')  # 专门添加分隔符的add
        token_drop_transform = TokenDropTransform()
        char_drop_transform = CharDropTransform()
        char_shape_transform = ShapeTransform()

        phonetic_transform = PhoneticTransform()
        phonetic_firstletter_transform = PhoneticTransform(first_letter=True)
        radical_transform = RadicalTransform('attacker/data/chaizi/chaizi-jt.txt', max_radicals_lengths=2)
        pronunciation_transform = PronunciationTransform('attacker/data/chaizi/hanzi.txt', N=50)

        same_radical_transform = SimpleSameRadicalTransform('attacker/data/chaizi/chaizi-jt.txt', max_radicals_lengths=2)

        hxw_transform = HuoXingWenTransform()

        phonetic_char_swap_transform = SequentialModel([phonetic_transform, char_swap_transform])
        phonetic_char_drop_transform = SequentialModel([phonetic_transform, char_drop_transform])
        phonetic_add_sep_transform = SequentialModel([phonetic_transform, add_sep_transform])
        phonetic_char_shape_transform = SequentialModel([phonetic_transform, char_shape_transform])

        hxw_radical_transform = SequentialModel([hxw_transform, radical_transform])
        radical_chardrop_transform = SequentialModel([radical_transform, char_drop_transform])
        hxw_radical_chardroptransform = SequentialModel(
        [hxw_transform, radical_transform, char_drop_transform])


        self.pinyin_transforms = [
        phonetic_transform,      # 拼音转pinyin
        phonetic_char_swap_transform,
        phonetic_char_drop_transform,
        phonetic_add_sep_transform,
        phonetic_char_shape_transform,
        pronunciation_transform      # 同音字转换
        ]

        self.zixing_transforms = [
        hxw_transform,
        radical_transform,
        hxw_radical_transform,
        same_radical_transform,
        radical_chardrop_transform,
        hxw_radical_chardroptransform
        ]

        self.normal_transforms = [
        random_replace_transform,
        add_transform,
        # token_drop_transform,
        token_swap_transform
        ]
        self.all_transforms = [self.pinyin_transforms, self.zixing_transforms, self.normal_transforms]
        self.all_p = [0.4, 0.4, 0.2]

    # def _append_transformed_tokens(self, transformed_tokens):
    #     if not transformed_tokens:
    #         return
    #     preprocessed_transform_text = ''.join(transformed_tokens)
    #     if preprocessed_transform_text in self.generated:
    #         return
    #     self.generated.append(preprocessed_transform_text)    

    def attack(self, raw_text, idx):
        tokens = [token for token in raw_text]
        #for idx in idxs:
        # transform_class = np.random.choice(self.all_transforms, 1, p=self.all_p)[0]
        # transform = np.random.choice(transform_class, 1)[0]
        transform_class = random.choices(self.all_transforms, self.all_p)[0]
        transform = random.choice(transform_class)
        trans_tokens = transform(tokens, idx)
        
        return ''.join(trans_tokens) if trans_tokens else ''.join(tokens)


"""
if __name__ == '__main__':

    gpt2_model_path = "/root/Data_Building/gpt2/pytorch_model.bin"
    gpt2_tokenizer_path = "/root/Data_Building/gpt2/"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"start load model")
    #gpt_model = torch.load(gpt2_model_path, map_location=device)
    gpt_model = GPT2LMHeadModel.from_pretrained(gpt2_tokenizer_path).to(device)
    logger.info(f"loaded model")
    gpt_tokenizer = BertTokenizer.from_pretrained(gpt2_tokenizer_path)
    gpt_model.eval()

    attack_numbers = 2
    obs_attacker = DataBuilder()
 #   obs_attacker.attack("你好你在干嘛你最近在吃什么呀")
    file_path = "gzh_data/test"
    out_path = "gzh_data/attackers.txt"
    f_in = open(file_path, "r", encoding="utf8")
    f_out = open(out_path, "w", encoding="utf8")

    for line in f_in:
        line_info = line.strip().split("\t")
        cur_texts = re.split("[。？！；]", line_info[-1])
        for i in range(len(cur_texts)):
            cur_sen = cur_texts[i]
            if len(cur_sen) <= 5:
                continue
            out_list = find_important_words_by_gpt2(cur_sen, gpt_model, gpt_tokenizer, device)
            out_list = out_list[:attack_numbers]
            indexs = [li[1]-1 for li in out_list]
            indexs.sort()
            out1 = obs_attacker.attack(cur_sen[:indexs[0]+1], indexs[0])
            out2 = obs_attacker.attack(cur_sen[indexs[0]+1:], indexs[1]-indexs[0])
            f_out.write(f"{cur_sen}\t{out1}{out2}\n")

"""