import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from tqdm import tqdm
from tensorboardX import SummaryWriter

from model import Model
from dataset import NewsDataset

import argparse

import os

def main(args):

    num_epochs = args.num_epochs
    save_model_epoch_num = args.save_model_epoch_num
    batch_size = args.batch_size
    output_size = args.output_size
    device_ids = args.device_ids
    use_cuda = args.use_cuda

    model = Model(title_embedding_size=args.title_embedding_size,
                            max_seq_len=args.max_seq_len,
                            output_size=output_size,
                            bert_pretrained_name=args.bert_pretrained_name_for_weight)

    if use_cuda and len(device_ids) != 0:
        model = model.cuda(device_ids[0])
        model = nn.parallel.DataParallel(model, device_ids=device_ids)
        model_module = model.module
    else:
        model_module = model

    if args.checkpoint_path is not None:
        if use_cuda and len(device_ids) != 0:
            model_module.load_state_dict(torch.load(args.checkpoint_path))
        else:
            model_module.load_state_dict(torch.load(args.checkpoint_path, map_location=torch.device('cpu')))
    #else:
    #    if args.tag_embedding_file is not None:
    #        model_module.tag_embedding.from_pretrained_weight(tag_embedding_file=args.tag_embedding_file)

    train_dataset = NewsDataset(real_news_path='data/True.csv', fake_news_path='data/Fake.csv')
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=args.shuffle, num_workers=args.num_workers, drop_last=True)
    test_dataset = NewsDataset(real_news_path='data/True.csv', fake_news_path='data/Fake.csv')
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=args.num_workers, drop_last=True)

    ce_loss = nn.CrossEntropyLoss()

    optims = {
        "title_embedding": torch.optim.Adam(model_module.title_embedding.parameters(), lr=args.lr_for_title_embedding),
        "fc": torch.optim.Adam(model_module.fc.parameters(), lr=args.lr_for_fc)
    }

    summary = SummaryWriter()


    print(args)


    train_global_iter = 0
    test_global_iter = 0
    for i_epoch in range(1, num_epochs+1):
        #################################
        ##         train epoch         ##
        #################################
        model.train()

        train_bar = tqdm(train_data_loader)
        running_results = {'batch_sizes': 0, 'loss': 0, 'total_num': 0, 'acc_num': 0}
        for data in train_bar:
            running_results['batch_sizes'] += 1
            train_global_iter += 1

            indexed_tokens, segments_ids, label = data
            batch_size_t = indexed_tokens.shape[0]

            if use_cuda and len(device_ids) != 0:
                indexed_tokens = indexed_tokens.cuda(device_ids[0])
                segments_ids = segments_ids.cuda(device_ids[0])
                label = label.cuda(device_ids[0])

            output = model(indexed_tokens.detach(), segments_ids.detach())

            predict = torch.argmax(output, dim=1)
            acc_num = torch.eq(predict.detach().cpu(), label.detach().cpu()).sum()

            loss = ce_loss(output, label)

            summary.add_scalar(tag='train_iter_loss', scalar_value=loss,
                               global_step=train_global_iter)

            optims["fc"].zero_grad()
            optims["title_embedding"].zero_grad()
            loss.backward()
            optims["fc"].step()
            optims["title_embedding"].step()

            running_results['total_num'] += batch_size_t
            running_results['acc_num'] += acc_num.item()
            running_results['loss'] += loss.item()

            train_bar.set_description(desc='Train [%d/%d] Loss: %.4f   ACC: %.4f'%
                (i_epoch, num_epochs,
                 running_results['loss'] / running_results['batch_sizes'],
                 running_results['acc_num'] / running_results['total_num'])
            )

        if i_epoch % save_model_epoch_num == 0:
            torch.save(model_module.state_dict(), os.path.join(args.save_path, 'epoch_%d.pth' % i_epoch))

        #################################
        ##          test epoch         ##
        #################################
        if args.val:
            model.eval()

            with torch.no_grad():

                test_bar = tqdm(test_data_loader)
                running_results = {'batch_sizes': 0, 'loss': 0, 'total_num': 0, 'acc_num': 0}
                for data in test_bar:

                    running_results['batch_sizes'] += 1
                    test_global_iter += 1

                    indexed_tokens, segments_ids, label = data
                    batch_size_t = indexed_tokens.shape[0]

                    if use_cuda and len(device_ids) != 0:
                        indexed_tokens = indexed_tokens.cuda(device_ids[0])
                        segments_ids = segments_ids.cuda(device_ids[0])
                        label = label.cuda(device_ids[0])

                    output = model(indexed_tokens.detach(), segments_ids.detach())

                    predict = torch.argmax(output, dim=1)
                    acc_num = torch.eq(predict.detach().cpu(), label.detach().cpu()).sum()

                    loss = ce_loss(output, label)

                    summary.add_scalar(tag='test_iter_loss', scalar_value=loss,
                                       global_step=test_global_iter)

                    running_results['total_num'] += batch_size_t
                    running_results['acc_num'] += acc_num.item()
                    running_results['loss'] += loss.item()

                    test_bar.set_description(desc='Test [%d/%d] Loss: %.4f   ACC: %.4f' %
                                               (i_epoch, num_epochs,
                                                running_results['loss'] / running_results['batch_sizes'],
                                                running_results['acc_num'] / running_results['total_num'])
                                          )



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train')
    ### model ###
    parser.add_argument('--output_size', default=2, type=int)
    parser.add_argument('--checkpoint_path', type=str)
    parser.add_argument('--val', default=True, type=bool)
    # title embedding
    parser.add_argument('--title_embedding_size', default=768, type=int)
    parser.add_argument('--max_seq_len', default=128, type=int)
    parser.add_argument('--bert_pretrained_name_for_weight', default='bert-base-uncased', type=str)
    parser.add_argument('--bert_pretrained_name_for_tokenizer', default='bert-base-uncased', type=str)

    ### dataset ###
    parser.add_argument('--pad_tag', default='[PAD]', type=str)
    parser.add_argument('--shuffle', default=True, type=bool)

    ### train ###
    parser.add_argument('--num_epochs', default=50, type=int)
    parser.add_argument('--save_model_epoch_num', default=1, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--use_cuda', default=True, type=bool)
    parser.add_argument('--device_ids', nargs='+', default=[], type=int)
    parser.add_argument('--save_path', default='./epochs/', type=str)


    parser.add_argument('--lr_for_title_embedding', default=0.00005, type=float)
    parser.add_argument('--lr_for_fc', default=0.001, type=float)


    args = parser.parse_args()

    main(args)
