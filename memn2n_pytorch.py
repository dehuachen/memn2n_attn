import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim



class MemN2NDialog(nn.Module):
    """docstring for MemN2NDialog"""

    def __init__(self, batch_size, vocab_size, candidates_size, sentence_size, embedding_size,
                 candidates_vec,
                 hops=3,
                 max_grad_norm=40.0,
                 nonlin=None,
                 optimizer=optim.Adam,
                 name='MemN2NDialog',
                 task_id=1):
        super(MemN2NDialog, self).__init__()

        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.candidates_size = candidates_size
        self.sentence_size = sentence_size
        self.embedding_size = embedding_size
        self.hops = hops
        self.max_grad_norm = max_grad_norm
        self.nonlin = nonlin
        self.name = name
        self.candidates = candidates_vec

        self.embed_A = nn.Embedding(self.vocab_size, self.embedding_size)
        self.linear_H = nn.Linear(self.embedding_size, self.embedding_size)
        self.embed_W = nn.Embedding(self.vocab_size, self.embedding_size)

        self.l1 = nn.Linear(self.embedding_size * 2, 1)

        self.softmax = nn.Softmax()
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        # weights initialization
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Embedding):
                m.weight.data.normal_(0, 0.1)
            if isinstance(m, nn.Embedding):
                m.weight.data[0].zero_()

        self.optimizer = optimizer(self.parameters(), lr=1e-2)

    def forward(self, stories, query):
        return self.inference(stories, query)

    def inference(self, stories, query):

        bs = query.size(0)

        # embed query
        query_emb = self.embed_A(query)
        query_emb_sum = torch.sum(query_emb, 1)
        u = [query_emb_sum]

        for _ in range(self.hops):
            stories_unbound = torch.unbind(stories, 1)
            embed_stories = [self.embed_A(story) for story in stories_unbound]
            embed_stories = torch.stack(embed_stories, 1)
            embed_stories_sum = torch.sum(embed_stories, 2)

            # get attention
            u_temp = torch.transpose(torch.unsqueeze(u[-1], -1), 1, 2)
            attention = torch.sum(embed_stories_sum * u_temp, 2)
            attention = self.softmax(attention)

            attention = torch.unsqueeze(attention, -1)
            attn_stories = torch.sum(attention * embed_stories_sum, 1)

            # output = self.linear_H(torch.cat([u[-1], attn_stories], 1))
            new_u = self.linear_H(u[-1]) + attn_stories

            u.append(new_u)

        candidates_emb = self.embed_W(self.candidates)
        candidates_emb_sum = torch.sum(candidates_emb, 1)
        candidates_emb_sum = candidates_emb_sum.unsqueeze(0).repeat(bs, 1, 1)
        new_u_n = new_u.unsqueeze(1).repeat(1, self.candidates_size, 1)

        x = torch.cat([candidates_emb_sum, new_u_n], 2)
        x = self.l1(x.view(-1, self.embedding_size * 2))
        x = self.softmax(x)
        x = x.view(bs, -1, 1)
        x = candidates_emb_sum * x
        x = torch.transpose(x, 1, 2)
        output = torch.bmm(new_u.unsqueeze(1), x).squeeze()
        # output = torch.mm(new_u, torch.transpose(candidates_emb_sum, 0, 1))

        return output

    def batch_fit(self, stories, query, answers):
        self.train()
        # calculate loss
        logits = self.forward(stories, query)
        cross_entropy = self.cross_entropy_loss(logits, answers)
        loss = torch.sum(cross_entropy)

        self.optimize(loss)
        return loss

    def optimize(self, loss):
        # calculate and apply grads
        self.optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm(self.parameters(), self.max_grad_norm)

        # weights initialization
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                m.weight.grad.data[0].zero_()

        self.optimizer.step()

    def predict(self, stories, query):
        self.eval()
        # calculate loss
        logits = self.forward(stories, query)
        _, preds = torch.max(logits, 1)

        return preds

