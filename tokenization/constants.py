
SEQUENCE_BOS_TOKEN = 0
SEQUENCE_PAD_TOKEN = 1
SEQUENCE_EOS_TOKEN = 2
SEQUENCE_UNK_TOKEN = 3
SEQUENCE_MASK_TOKEN = 29
VOCAB_SIZE = 30

SEQUENCE_BOS_STR = "<cls>"
SEQUENCE_PAD_STR = "<pad>"
SEQUENCE_EOS_STR = "<eos>"
SEQUENCE_UNK_STR = "<unk>"
MASK_STR_SHORT = "_"
SEQUENCE_MASK_STR = "<mask>"

SEQUENCE_VOCAB = [
    "<cls>", "<pad>", "<eos>", "<unk>",
    "T", "X", "V", "F", "Q", "I", "P", "3", "H", "S", "M", "K",
    "G", "N", "2", "R", "L", "E", "C", "Y", "A", "D", "B", "W",
    "_",
    "<mask>",
]

INTENSITY_BOUNDARIES = [4.0, 9.0, 13.0]
# INTENSITY_BOUNDARIES = [4.0]
INTENSITY_MASK_TOKEN = len(INTENSITY_BOUNDARIES)+1
NUM_INTENSITY_BIN = INTENSITY_MASK_TOKEN+1