"""
同源操作，即通过drop、swap等操作实现数据变形
"""
import re
import numpy as np
import random
from .transform import Transform



class RandomReplaceTransform(Transform):
  def __init__(self, chinese_chars_file):
    super().__init__()
    with open(chinese_chars_file, encoding='utf-8') as f:
        processed_char_set = set()
        for char in f:
            char = char.strip()
            if char in processed_char_set:
                continue
            processed_char_set.add(char)  
        self.pool = list(processed_char_set)

  def __call__(self, tokens, idx):
    cur_char_pool = list(set(''.join(tokens)))
    if not cur_char_pool:
      return None
    noise = random.choice(cur_char_pool + self.pool)

    new_tokens = tokens[:]
    new_tokens[idx] = noise
    if self.debug:
      self.transformed_tokens.append(new_tokens)
    return new_tokens


class TokenSwapTransform(Transform):
  def __init__(self, range=(1, 2)):
    """
    交换tokens[idx]和tokens[idx+offset]
    """
    super().__init__()
    self.range = range

  def __call__(self, tokens, idx):
    if len(tokens) <= 4:
      return tokens[:]

    range = self.range
    offset = np.random.randint(range[0], range[1] + 1, size=1)[0]
    if np.random.random() <= 0.5:
      offset *= -1
    while idx + offset < 0 or idx + offset >= len(tokens):
      offset = np.random.randint(range[0], range[1] + 1, size=1)[0]
      if np.random.random() <= 0.5:
        offset *= -1

    new_tokens = tokens[:]
    new_tokens[idx], new_tokens[idx + offset] = new_tokens[idx + offset], new_tokens[idx]

    if self.debug:
      self.transformed_tokens.append(new_tokens)
    return new_tokens


class CharSwapTransform(Transform):
  def __init__(self):
    super().__init__()

  def __call__(self, tokens, idx):
    target_token = tokens[idx]
    new_token = self._transform(target_token)
    if target_token == new_token:
      return None

    new_tokens = tokens[:]
    new_tokens[idx] = new_token
    if self.debug:
      self.transformed_tokens.append(new_tokens)
    return new_tokens

  def _transform(self, target_token):
    if len(target_token) < 4:
      return target_token

    middle_chars = list(target_token[1:-1])
    if len(set(middle_chars)) > 1:
      while middle_chars == list(target_token[1:-1]):
        random.shuffle(middle_chars)
    new_token = target_token[0] + ''.join(middle_chars) + target_token[-1]
    return new_token


class TokenDropTransform(Transform):
  def __init__(self):
    super().__init__()

  def __call__(self, tokens, idx):
    new_tokens = tokens[:idx] + tokens[idx + 1:]
    if self.debug:
      self.transformed_tokens.append(new_tokens)
    return new_tokens


class CharDropTransform(Transform):
  def __init__(self):
    super().__init__()

  def __call__(self, tokens, idx):
    target_token = tokens[idx]
    if len(target_token) > 1:
      random_idx = random.randint(0, len(target_token) - 1)
      new_token = target_token[:random_idx] + target_token[random_idx + 1:]

      new_tokens = tokens[:idx] + [new_token] + tokens[idx + 1:]
    else:
      new_tokens = tokens[:idx] + tokens[idx + 1:]
    if self.debug:
      self.transformed_tokens.append(new_tokens)
    return new_tokens


class AddTransform(Transform):
  def __init__(self, chinese_chars_file, pool=None):
    super().__init__()
    if pool is not None:
      self.pool = list(pool)
    else:
      with open(chinese_chars_file, encoding='utf-8') as f:
        processed_char_set = set()
        for char in f:
            char = char.strip()
            if char in processed_char_set:
                continue
            processed_char_set.add(char)  
      self.pool = list(processed_char_set)

  def __call__(self, tokens, idx):
    # cur_char_pool = list(set(''.join(tokens)))
    noise = random.choice(self.pool)
    new_tokens = tokens[:idx] + [noise] + tokens[idx:]
    if self.debug:
      self.transformed_tokens.append(new_tokens)
    return new_tokens

  def multi_ptr_trans(self, tokens, indices):
    new_tokens = tokens[:]
    noise = random.choice(self.pool)  # 用同一个噪声，克服jaccard指标上的劣势
    for idx in indices:  # 默认已经是降序排列
      new_tokens.insert(idx, noise)
    return new_tokens
