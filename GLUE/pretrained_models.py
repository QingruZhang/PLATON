from transformers import BertConfig, BertModel, BertTokenizer
from transformers import XLNetConfig, XLNetModel, XLNetTokenizer
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer
from transformers import AlbertConfig, AlbertModel, AlbertTokenizer
from transformers import XLMRobertaConfig, XLMRobertaModel, XLMRobertaTokenizer
from transformers import ElectraConfig, ElectraModel, ElectraTokenizer
from transformers import T5Config, T5EncoderModel, T5Tokenizer
from transformers import DebertaConfig, DebertaModel, DebertaTokenizer
from module.san_model import SanModel
from module.modeling_bert_masked import MaskedBertModel
from module.modeling_roberta_masked import MaskedRobertaModel
MODEL_CLASSES = {
    "bert": (BertConfig, BertModel, BertTokenizer),
    "xlnet": (XLNetConfig, XLNetModel, XLNetTokenizer),
    "roberta": (RobertaConfig, RobertaModel, RobertaTokenizer),
    "albert": (AlbertConfig, AlbertModel, AlbertTokenizer),
    "xlm": (XLMRobertaConfig, XLMRobertaModel, XLMRobertaTokenizer),
    "san": (BertConfig, SanModel, BertTokenizer),
    "electra": (ElectraConfig, ElectraModel, ElectraTokenizer),
    "t5": (T5Config, T5EncoderModel, T5Tokenizer),
    "deberta": (DebertaConfig, DebertaModel, DebertaTokenizer),
    "mbert": (BertConfig, MaskedBertModel, BertTokenizer),
    "mroberta": (RobertaConfig, MaskedRobertaModel, RobertaTokenizer),
}
