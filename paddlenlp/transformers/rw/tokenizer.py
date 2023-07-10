# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from paddlenlp.transformers import (
    BasicTokenizer,
    WordpieceTokenizer,
)
from paddlenlp.transformers.gpt.tokenizer import GPTTokenizer

__all__ = ["RWTokenizer"]


class RWTokenizer(GPTTokenizer):
    """
    Constructs a RW tokenizer. It uses a basic tokenizer to do punctuation
    splitting, lower casing and so on, and follows a WordPiece tokenizer to
    tokenize as subwords.

    Args:
        vocab_file (str):
            The vocabulary file path (ends with '.txt') required to instantiate
            a `WordpieceTokenizer`.
        do_lower_case (bool):
            Whether or not to lowercase the input when tokenizing.
            Defaults to`True`.
        unk_token (str):
            A special token representing the *unknown (out-of-vocabulary)* token.
            An unknown token is set to be `unk_token` inorder to be converted to an ID.
            Defaults to "[UNK]".
        sep_token (str):
            A special token separating two different sentences in the same input.
            Defaults to "[SEP]".
        pad_token (str):
            A special token used to make arrays of tokens the same size for batching purposes.
            Defaults to "[PAD]".
        cls_token (str):
            A special token used for sequence classification. It is the last token
            of the sequence when built with special tokens. Defaults to "[CLS]".
        mask_token (str):
            A special token representing a masked token. This is the token used
            in the masked language modeling task which the model tries to predict the original unmasked ones.
            Defaults to "[MASK]".

    Examples:
        .. code-block::

            from paddlenlp.transformers import RWTokenizer
            
            tokenizer = RWTokenizer.from_pretrained('nezha-base-chinese')

            inputs = tokenizer('欢迎使用百度飞桨！')
            print(inputs)

            '''
            {'input_ids': [101, 3614, 6816, 886, 4500, 4636, 2428, 7607, 3444, 8013, 102],
            'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
            '''

    """

    resource_files_names = {"vocab_file": "vocab.json", "merges_file": "merges.txt"}  # for save_pretrained
    model_input_names = ["input_ids"]
    
    pretrained_resource_files_map = {
        "vocab_file": {
            "falcon-7b": "https://bj.bcebos.com/paddlenlp/models/community/tiiuae/falcon-7b/vocab.json",
        },
        "merges_file": {
            "falcon-7b": "https://bj.bcebos.com/paddlenlp/models/community/tiiuae/falcon-7b/merges.txt",
        }
    }
    padding_side = "right"

    def __init__(
        self,
        vocab_file,
        merges_file,
        **kwargs  # The token of newline.
    ):
        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'. To load the "
                "vocabulary from a pretrained model please use "
                "`tokenizer = RWTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`".format(vocab_file)
            )
        super().__init__(vocab_file, merges_file, **kwargs)
