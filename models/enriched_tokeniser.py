import nltk
import spacy
# Space module import
import en_core_web_md
from enum import Enum

import torch.nn.functional as F
import torch

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def get_spacy_module():
  return en_core_web_md.load()

nlp = get_spacy_module()

def get_sentence_tokens(s, display = False):
  doc = nlp(s)
  sentence_tokens = []
  if display:
    spacy.displacy.render(doc, style="dep", jupyter=True)
  for token in doc:
    data = {
      "token_id": token.i,
      "token_text": token.text,
      "token_connection_ids": token.head.i,
      "token_left_edge": token.idx,
      "token_right_edge": token.idx + len(token.text),
      "token_boundaries": (token.idx, token.idx + len(token.text)),
      "token_pos_tag": token.tag_,
    }

    sentence_tokens.append(data)
  return sentence_tokens

class TokeniserType(Enum):
    ONE_SENTENCE = 1
    TWO_SENTENCES = 2

class BaseEnrichedTokeniser:
  def __init__(self, tokeniser):
    self._tokeniser = tokeniser
    self.feature_key = None
    self.type = TokeniserType.ONE_SENTENCE

  def combine_transformer_and_sentence_features(self, transformer_tokens, sentence_features):
    for i in range(len(transformer_tokens)):
      transformer_tokens[i][self.feature_key] = None
      for j in range(len(sentence_features)):
        t_boundaries = transformer_tokens[i]['boundaries']
        s_boundaries = sentence_features[j]['boundaries']
        left = max(t_boundaries[0], s_boundaries[0])
        right = min(t_boundaries[1], s_boundaries[1])
        if left < right:
          transformer_tokens[i][self.feature_key] = sentence_features[j][self.feature_key]

      assert(transformer_tokens[i][self.feature_key] is not None)

    return transformer_tokens

  def get_feature(self, s):
    raise Exception("Not implemented")

  def enrich_tokens(self, s):
    t_tokens = self.get_transformer_sentence_tokens(s)
    feature = self.get_feature(s)
    combined_features = self.combine_transformer_and_sentence_features(t_tokens, feature)
    return combined_features

  """
  Return list of tokens that will be used in model
  [('boundaries':(int, int), 'input_id':int, 'index':int]
  """
  def get_transformer_sentence_tokens(self, s, verbose = False):
      encoded = self._tokeniser.batch_encode_plus([s], return_offsets_mapping=True, add_special_tokens=False)
      split_tokens = []
      for i in range(len(encoded['input_ids'][0])):
        split_tokens.append(
            {
                'index': i,
                'input_id': encoded['input_ids'][0][i],
                'boundaries': encoded['offset_mapping'][0][i],
            }
        )
      return split_tokens

  def post_processing(self, v):
    return v

class PosTagEnrichedTokeniser(BaseEnrichedTokeniser):

  def __init__(self, tokeniser):
    super().__init__(tokeniser)
    self.feature_key = 'pos_tag'

  def get_feature(self, s):
    """
    Return list of [('boundaries':(int, int), 'feature':string)]
    """
    all_tokens = get_sentence_tokens(s)
    return [{'boundaries':token['token_boundaries'], 'pos_tag': token['token_pos_tag']} for token in all_tokens]


class PosTagIdEnrichedTokeniser(BaseEnrichedTokeniser):
  def __init__(self, tokeniser):
    super().__init__(tokeniser)
    self.feature_key = 'pos_tag_ids'
    self.pos_tag_to_id_map = PosTagIdEnrichedTokeniser.load_pos_tag_value_to_idx()

  @staticmethod
  def load_pos_tag_value_to_idx():
  # Computed based on test part of MRPC dataset.
    return {
        '$': 1,
        "''": 2,
        '(': 3,
        ')': 4,
        ',': 5,
        '.': 6,
        ':': 7,
        'CC': 8,
        'CD': 9,
        'DT': 10,
        'EX': 11,
        'FW': 12,
        'IN': 13,
        'JJ': 14,
        'JJR': 15,
        'JJS': 16,
        'LS': 17,
        'MD': 18,
        'NA': 19,
        'NN': 20,
        'NNP': 21,
        'NNPS': 22,
        'NNS': 23,
        'PRP': 24,
        'PRP$': 25,
        'RB': 26,
        'RBR': 27,
        'SYM': 28,
        'TO': 29,
        'UH': 30,
        'UNKNOWN': 31,
        'VB': 32,
        'VBD': 33,
        'VBG': 34,
        'VBN': 35,
        'VBP': 36,
        'VBZ': 37,
        'WDT': 38,
        'WP': 39,
        'WP$': 40,
        'WRB': 41,
        '``': 42
      }

  def get_feature(self, s):
    """
    Return list of [('boundaries':(int, int), 'feature':string)]
    """
    all_tokens = get_sentence_tokens(s)
    def get_id(pos_tag):
      if pos_tag in self.pos_tag_to_id_map:
        return self.pos_tag_to_id_map[pos_tag]
      return self.pos_tag_to_id_map['UNKNOWN']
    return [{'boundaries':token['token_boundaries'], self.feature_key: get_id(token['token_pos_tag'])} for token in all_tokens]

  def post_processing(self, s1_s2_feature):
    token_mapping = self.get_special_token_mapping()
    def replace_token_id(token_id, m):
      if token_id in m.keys():
        return m[token_id]
      return token_id
    _s1_s2_feature = [replace_token_id(x, token_mapping) for x in s1_s2_feature]
    return _s1_s2_feature

  def get_special_token_mapping(self):
    return {
      self._tokeniser.cls_token_id: self.pos_tag_to_id_map['NA'],
      self._tokeniser.sep_token_id: self.pos_tag_to_id_map['NA'],
    }


class AttentionEnhencerDummyEnrichedTokeniser(BaseEnrichedTokeniser):
  def __init__(self, tokeniser):
    super().__init__(tokeniser)
    self.feature_key = 'attention_enhencer_dummy'
    self.type = TokeniserType.TWO_SENTENCES

  def enrich_tokens(self, s1, s2,  padding, truncation, max_length, _config):
    # Dummy matrix will have 1 for all non padding elements.

    dummy = self._tokeniser(s1, s2, truncation=truncation, max_length=max_length, padding=padding)

    first_padding_0 = dummy['input_ids'].index(self._tokeniser.pad_token_id)
    source = torch.full((first_padding_0,first_padding_0), 1.)
    pad_distance =  max_length - first_padding_0

    result = F.pad(input=source, pad=(0, pad_distance, 0, pad_distance), mode='constant', value=0.)
    return result

  def get_feature(self, s):
    all_tokens = get_sentence_tokens(s)
    return [
      {
        'boundaries':token['token_boundaries'],
        'token_connection_ids': token['token_connection_ids'],
        'token_id': token['token_id']
      } for token in all_tokens
    ]


class AttentionEnhencerRandEnrichedTokenise(BaseEnrichedTokeniser):
  def __init__(self, tokeniser):
    super().__init__(tokeniser)
    self.feature_key = 'attention_enhencer_rand'
    self.type = TokeniserType.TWO_SENTENCES

  def enrich_tokens(self, s1, s2,  padding, truncation, max_length, _config):
    # Dummy matrix will have 1 for all non padding elements.

    dummy = self._tokeniser(s1, s2, truncation=truncation, max_length=max_length, padding=padding)

    first_padding_0 = dummy['input_ids'].index(self._tokeniser.pad_token_id)
    source = torch.rand((first_padding_0,first_padding_0))
    pad_distance =  max_length - first_padding_0

    result = F.pad(input=source, pad=(0, pad_distance, 0, pad_distance), mode='constant', value=0.)
    return result

  def get_feature(self, s):
    all_tokens = get_sentence_tokens(s)
    return [
      {
        'boundaries':token['token_boundaries'],
        'token_connection_ids': token['token_connection_ids'],
        'token_id': token['token_id']
      } for token in all_tokens
    ]

class AttentionEnhencerOneEnrichedTokenise(BaseEnrichedTokeniser):
  def __init__(self, tokeniser):
    super().__init__(tokeniser)
    self.feature_key = 'attention_enhencer'
    self.type = TokeniserType.TWO_SENTENCES

  def enrich_tokens(self, s1, s2,  padding, truncation, max_length, _config):
    # Dummy matrix will have 1 for all elements.
    source = torch.full((max_length,max_length), 1.)

    return source

class AttentionEnhencerDependancyTreeEnrichedTokenise(BaseEnrichedTokeniser):
  def __init__(self, tokeniser):
    super().__init__(tokeniser)
    self.feature_key = 'attention_enhencer'
    self.type = TokeniserType.TWO_SENTENCES

  def enrich_tokens(self, s1, s2,  padding, truncation, max_length, config):
    if 'att_dep_tree_pad_value' in config:
      pad_value = config['att_dep_tree_pad_value']
    else:
      pad_value = 0.

    data = self._tokeniser(s1, s2, truncation=truncation, max_length=max_length, padding=padding)
    first_padding_0 = data['input_ids'].index(self._tokeniser.pad_token_id)
    source = torch.full((first_padding_0,first_padding_0), 1.)
    pad_distance =  max_length - first_padding_0
    base_table = F.pad(input=source, pad=(0, pad_distance, 0, pad_distance), mode='constant', value=pad_value)
    # Here we have table
    # [1, ...1, 0, ..0]
    # [...............]
    # [1, ...1, 0, ..0]
    # [...............]
    # [0, ...0, 0, ..0]

    # First sentence start
    s1_start_inx = 1
    sep_inx = data['input_ids'].index(self._tokeniser.sep_token_id)
    s1_end_inx = sep_inx - 1
    s2_start_inx = sep_inx + 1
    s2_end_inx = first_padding_0 - 2

    # print(f"First start {s1_start_inx} end {s1_end_inx}")
    # print(f"Second start {s2_start_inx} end {s2_end_inx}")

    # for i in range(s1_end_inx - s1_start_inx + 1):
    #   for j in range(s1_end_inx - s1_start_inx + 1):
    #     base_table[s1_start_inx + i][s1_start_inx + j] = 2.

    # for i in range(s2_end_inx - s2_start_inx + 1):
    #   for j in range(s2_end_inx - s2_start_inx + 1):
    #     base_table[s2_start_inx + i][s2_start_inx + j] = 3.

    s1_tokens = self.get_transformer_sentence_tokens(s1)
    s2_tokens = self.get_transformer_sentence_tokens(s2)

    s1_feature = self.get_feature(s1)
    s2_feature = self.get_feature(s2)

    def boundaries_match(transformer_token_boundary, sentence_feature_boundary):
      left = max(transformer_token_boundary[0], sentence_feature_boundary[0])
      right = min(transformer_token_boundary[1], sentence_feature_boundary[1])
      return left < right

    def build_token_map(tokens, features):
      # Feature token id => [list of token ids]
      m = {}
      for i in range(len(features)):
        m[i] = []
        for j in range(len(tokens)):
          feature_boundary = features[i]['boundaries']
          token_boundary = tokens[j]['boundaries']
          if boundaries_match(token_boundary, feature_boundary):
            # we need to add pair
            m[i].append(j)
      return m

    val = config['att_dep_tree_value']

    def update_base_table(table, tokens, features, max_len, offset):
      token_map = build_token_map(tokens, features)
      for i in range(len(features)):
        feature = features[i]
        connected_node_id = feature['token_connection_ids']
        feature_connected = features[connected_node_id]
        tokens = token_map[i]
        connected_tokens = token_map[connected_node_id]
        for _x in tokens:
          for _y in connected_tokens:
            if _x < max_len and _y < max_len:
              table[offset + _x][offset + _y] = val
              table[offset + _y][offset + _x] = val
      return table

    base_table = update_base_table(base_table, s1_tokens, s1_feature, s1_end_inx - s1_start_inx + 1, s1_start_inx)
    base_table = update_base_table(base_table, s2_tokens, s2_feature, s2_end_inx - s2_start_inx + 1, s2_start_inx)

    # number_of_attention_heads = 12
    # base_table = base_table[None, :, :].expand([number_of_attention_heads, max_length, max_length])
    return base_table

  def get_feature(self, s):
    all_tokens = get_sentence_tokens(s)
    return [
      {
        'boundaries':token['token_boundaries'],
        'token_connection_ids': token['token_connection_ids'],
        'token_id': token['token_id']
      } for token in all_tokens
    ]

class FinalTokeniser:
  def __init__(self, tokeniser, config):
    self._tokeniser = tokeniser
    self._config = config

  @staticmethod
  def _prepare_for_model(tokeniser, s1, s2, padding, truncation, max_length):
    return tokeniser.prepare_for_model(
      s1,
      s2,
      padding=padding,
      truncation=truncation,
      max_length=max_length,
    )['input_ids']

  def apply_tokenisers(self, s1, s2, tokeniser_list, padding, truncation, max_length):
    # Get base data first: input ids, token_ids, attention_mask
    encoded_base = self._tokeniser.batch_encode_plus([s1, s2], return_offsets_mapping=True, add_special_tokens=False)
    s1_input_ids = encoded_base['input_ids'][0]
    s2_input_ids = encoded_base['input_ids'][1]

    _prepare_for_model = FinalTokeniser._prepare_for_model

    s1_s2_input_ids = _prepare_for_model(self._tokeniser, s1_input_ids, s2_input_ids, padding, truncation, max_length)

    data = {
      'input_ids': s1_s2_input_ids,
    }

    for tokeniser in tokeniser_list:
      if tokeniser.type == TokeniserType.ONE_SENTENCE:
        enriched_data_s1 = tokeniser.enrich_tokens(s1)
        enriched_data_s2 = tokeniser.enrich_tokens(s2)

        _s1_input_ids = [x['input_id'] for x in enriched_data_s1]
        _s2_input_ids = [x['input_id'] for x in enriched_data_s2]

        assert(s1_input_ids == _s1_input_ids)
        assert(s2_input_ids == _s2_input_ids)

        s1_feature = [x[tokeniser.feature_key] for x in enriched_data_s1]
        s2_feature = [x[tokeniser.feature_key] for x in enriched_data_s2]

        s1_s2_feature = _prepare_for_model(self._tokeniser, s1_feature, s2_feature,  padding, truncation, max_length)
      else:
        s1_s2_feature = tokeniser.enrich_tokens(s1, s2,  padding, truncation, max_length, self._config)

      s1_s2_feature = tokeniser.post_processing(s1_s2_feature)
      data[tokeniser.feature_key] = s1_s2_feature
    return data

  def tokenise_everything(self, s1, s2, padding, truncation, max_length):
    if 'tokeniser_list' in self._config:
      tokeniser_list = self._config['tokeniser_list']
    else:
      tokeniser_list = ['dep']

    list_of_tokenisers = []
    if 'pos' in tokeniser_list:
      pos_tag_id_tokeniser = PosTagIdEnrichedTokeniser(self._tokeniser)
      list_of_tokenisers.append(pos_tag_id_tokeniser)
    if 'attention_dummy' in tokeniser_list:
      attention_dummy_tokeniser = AttentionEnhencerDummyEnrichedTokeniser(self._tokeniser)
      list_of_tokenisers.append(attention_dummy_tokeniser)

    if 'attention_dep' in tokeniser_list:
      attention_tokeniser_dep = AttentionEnhencerDependancyTreeEnrichedTokenise(self._tokeniser)
      list_of_tokenisers.append(attention_tokeniser_dep)

    if 'attention_one' in tokeniser_list:
      attention_tokeniser_one = AttentionEnhencerOneEnrichedTokenise(self._tokeniser)
      list_of_tokenisers.append(attention_tokeniser_one)

    return self.apply_tokenisers(
      s1,
      s2,
      list_of_tokenisers,
      padding,
      truncation,
      max_length
    )

def preprocess_dataset_final(examples, tokenizer, truncation, max_length, padding, config):
  basic_tokenizer_data = tokenizer(examples["sentence1"], examples["sentence2"], truncation=truncation, max_length=max_length, padding=padding)
  final_tokeniser = FinalTokeniser(tokeniser=tokenizer, config=config)
  enriched_data = final_tokeniser.tokenise_everything(examples["sentence1"], examples["sentence2"], truncation=truncation, max_length=max_length, padding=padding)
  assert(basic_tokenizer_data['input_ids'] == enriched_data['input_ids'])
  for feature, value in enriched_data.items():
    basic_tokenizer_data[feature] = value
  return basic_tokenizer_data