import torch
from transformers import BertTokenizer

from application.model.model_bert import Model

def preprocess_sentence(sentence):
    """Process sentence to get input in bert format（indexed_tokens, segments_ids）
    """
    # Process title, get index
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenized_text = tokenizer.tokenize(sentence)[:128 - 2]
    # get [PAD]
    if len(tokenized_text) < 128 - 1:
        tokenized_text.extend(['[PAD]' for _ in range(128 - len(tokenized_text) - 1)])
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    
    segments_ids = [0 for _ in range(len(indexed_tokens))]

    return torch.tensor(indexed_tokens), torch.tensor(segments_ids)

def predict_bert(text):

    checkpoint_path = "./application/model/brain/epoch_1.pth"
    devide = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(title_embedding_size=768,
                  max_seq_len=128,
                  output_size=2,
                  bert_pretrained_name="bert-base-uncased")

    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
    model.to(devide)

    model.eval()

    id2class = {
        0: "fake",
        1: "real"
    }

    with torch.no_grad():

        indexed_tokens, segments_ids = preprocess_sentence(text)

        indexed_tokens = indexed_tokens.unsqueeze(dim=0).to(devide)
        segments_ids = segments_ids.unsqueeze(dim=0).to(devide)

        output = model(indexed_tokens, segments_ids)

        predict = output.argmax()
        print("Input: %s" % text)
        print(id2class[predict.item()])

        return id2class[predict.item()]

if __name__ == "__main__":
    text = "WASHINGTON (Reuters) - The head of a conservative Republican faction in the U.S. Congress, who voted this month for a huge expansion of the national debt to pay for tax cuts, called himself a “fiscal conservative” on Sunday and urged budget restraint in 2018. In keeping with a sharp pivot under way among Republicans, U.S. Representative Mark Meadows, speaking on CBS’ “Face the Nation,” "

    result = predict(text)
