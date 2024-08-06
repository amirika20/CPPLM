import os
import sys
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import tokenization.constants as constants

class CPPTokenizer:
    def __init__(self, vocab=None):
        """
        Initialize the custom tokenizer.
        :param vocab: A list of tokens to use for tokenization.
        """
        if vocab is None:
            self.vocab = self.build_vocab()
        else:
            self.vocab = vocab
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}

    def build_vocab(self):
        """
        Build a custom vocabulary.
        :return: A list of tokens.
        """
        # Define your vocabulary here
        return constants.SEQUENCE_VOCAB

    def tokenize(self, sequence):
        """
        Tokenize the input sequence.
        :param sequence: The input sequence to tokenize.
        :return: A list of tokens.
        """
        # return list(sequence.replace("2","").replace("3","")) # TODO
        return list(sequence)

    def convert_tokens_to_ids(self, tokens):
        """
        Convert a list of tokens to their corresponding IDs.
        :param tokens: A list of tokens.
        :return: A list of token IDs.
        """
        return [self.token_to_id.get(token, constants.SEQUENCE_UNK_TOKEN) for token in tokens]

    def convert_ids_to_tokens(self, token_ids):
        """
        Convert a list of token IDs back to their corresponding tokens.
        :param token_ids: A list of token IDs.
        :return: A list of tokens.
        """
        return [self.id_to_token.get(id, constants.SEQUENCE_UNK_STR) for id in token_ids]

    def tokenize_sequence(self, sequence, max_length=128):
        """
        Tokenize the input sequence and convert to token IDs.
        :param sequence: The input sequence to tokenize.
        :param max_length: The maximum length of the tokenized sequence.
        :return: A dictionary containing the tokenized sequence and attention mask.
        """
        tokens = self.tokenize(sequence)
        token_ids = self.convert_tokens_to_ids(tokens)

        # Add special tokens and pad/truncate the sequence
        token_ids = [constants.SEQUENCE_BOS_TOKEN] + token_ids + [constants.SEQUENCE_EOS_TOKEN]
        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        else:
            token_ids += [constants.SEQUENCE_PAD_TOKEN] * (max_length - len(token_ids))

        padding_mask = [1 if id != constants.SEQUENCE_PAD_TOKEN else 0 for id in token_ids]

        return {
            "input_ids": torch.LongTensor(token_ids),
            "padding_mask": torch.LongTensor(padding_mask)
        }

    def tokenize_batch(self, sequences):
        tokenized_sequences = list(map(lambda x:self.tokenize_sequence(x), sequences))
        input_ids = torch.stack(list(map(lambda x:x['input_ids'], tokenized_sequences)))
        padding_mask = torch.stack(list(map(lambda x:x['padding_mask'], tokenized_sequences)))
        return input_ids, padding_mask


    def decode_tokens(self, token_ids):
        """
        Decode a list of token IDs back to a string.
        :param token_ids: The list of token IDs to decode.
        :return: The decoded string.
        """
        tokens = self.convert_ids_to_tokens(token_ids)
        return "".join(tokens).replace(constants.SEQUENCE_PAD_STR, "").replace(constants.SEQUENCE_BOS_STR, "").replace(constants.SEQUENCE_EOS_STR, "")

def main():
    # Example usage
    sequence = "LLIILRRRIRKQAHAHSK2KRVKAGYLLGKINLKALAALAKKIL3RQIKIWFQNRRMKWKK"
    tokenizer = CPPTokenizer()
    tokenized_output = tokenizer.tokenize_sequence(sequence)

    print("Tokenized sequence IDs:", tokenized_output['input_ids'])
    print("Attention mask:", tokenized_output['attention_mask'])

    decoded_sequence = tokenizer.decode_tokens(tokenized_output['input_ids'])
    print("Decoded sequence:", decoded_sequence)
    print(sequence==decoded_sequence)

if __name__ == "__main__":
    # main()
    text = "SDFSDFSDF2SDFASDF3ASDFSDAF"
    print(text.replace("2","").replace("3",""))
