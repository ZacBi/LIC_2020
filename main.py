import argparse
import logging

from pytorch_pretrained_bert import BertTokenizer, BertModel
from pytorch_pretrained_bert.modeling import BertConfig, BertForSequenceClassification

from feeder import feed
from trainer import train

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", default="data/train.json", type=str, help=None)
    parser.add_argument("--dev_file", default="data/dev.json", type=str, help=None)
    parser.add_argument("--test_file", default="data/test.json", type=str, help=None)
    parser.add_argument("--pretrained_model", default="bert-base-chinese", type=str, help=None)
    parser.add_argument("--learning_rate", default=5e-5, type=float, help=None)
    parser.add_argument("--train_batch_size", default=4, type=int, help=None)
    parser.add_argument("--test_batch_size", default=4, type=int, help=None)
    parser.add_argument("--num_train_epochs", default=200, type=int, help=None)
    parser.add_argument("--num_stop_train", default=200, type=int, help=None)
    parser.add_argument("--do_debug", default=False, type=bool, help=None)
    args = parser.parse_args()
 
    logger = logging.getLogger()
    logging.basicConfig(format = '%(asctime)s - %(message)s', datefmt = '%H:%M:%S', level=logging.INFO)

    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model)
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
    
    train_data, test_data, num_labels = feed(args,tokenizer)
    model = BertForSequenceClassification(config, num_labels).cuda()
    train(args, logger, tokenizer, model, train_data, test_data)

    logger.info("Done")

if __name__ == "__main__":
    main()
