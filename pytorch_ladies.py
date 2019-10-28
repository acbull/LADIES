#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from utils import *
from tqdm import tqdm
import argparse


# In[ ]:


import sys; sys.argv=['']; del sys
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


# In[ ]:


parser = argparse.ArgumentParser(description='Training GCN on Cora/CiteSeer/PubMed Datasets')

'''
    Dataset arguments
'''
parser.add_argument('--dataset', type=str, default='reddit',
                    help='Dataset name: Cora/CiteSeer/PubMed')
parser.add_argument('--nhid', type=int, default=256,
                    help='Hidden state dimension')
parser.add_argument('--epoch_num', type=int, default= 100,
                    help='Number of Epoch')
parser.add_argument('--pool_num', type=int, default= 5,
                    help='Number of Pool')
parser.add_argument('--batch_num', type=int, default= 10,
                    help='Maximum Batch Number')
parser.add_argument('--batch_size', type=int, default=512,
                    help='size of output node in a batch')
parser.add_argument('--n_layers', type=int, default=5,
                    help='Number of GCN layers')
parser.add_argument('--samp_num', type=int, default=64,
                    help='Number of sampled nodes per layer')
parser.add_argument('--sample_method', type=str, default='ladies',
                    help='Sampled Algorithms: ladies/fastgcn/full')
parser.add_argument('--cuda', type=int, default=-1,
                    help='Avaiable GPU ID')


# In[ ]:


args = parser.parse_args()


# In[ ]:


class GraphConvolution(nn.Module):
    def __init__(self, n_in, n_out, bias=True):
        super(GraphConvolution, self).__init__()
        self.n_in  = n_in
        self.n_out = n_out
        self.linear = nn.Linear(n_in,  n_out)
    def forward(self, x, adj):
        out = self.linear(x)
        return F.relu(torch.spmm(adj, out))


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, layers, dropout):
        super(GCN, self).__init__()
        self.layers = layers
        self.nhid = nhid
        self.gcs = nn.ModuleList()
        self.gcs.append(GraphConvolution(nfeat,  nhid))
        self.norms = nn.ModuleList()
        self.norms.append(BatchNorm(nhid))
        self.dropout = nn.Dropout(dropout)
        for i in range(layers-1):
            self.gcs.append(GraphConvolution(nhid,  nhid))
            self.norms.append(BatchNorm(nhid))
    def forward(self, x, adjs):
        for idx in range(len(self.gcs)):
            x = self.dropout(self.gcs[idx](x, adjs[idx]))
        return x
    def hier_forward(self, x, adjs):
        res = [x]
        for idx in range(len(self.gcs)):
            x = self.gcs[idx](x, adjs[idx])
            res += [x.detach()]
        return res  
    def cal_mamory(self, x, adjs):
        res = self.hier_forward(x, adjs)
        sz = 0
        for ri in res:
            sz += np.prod(np.array(ri.shape))
        for ai in adjs:
            sz += (ai.to_dense() > 0).sum().tolist()
        for pi in self.parameters():
            sz += np.prod(np.array(pi.shape))
        return sz * 32 / 1024 / 1024 / 8
    
class SuGCN(nn.Module):
    def __init__(self, encoder, num_classes, dropout, inp):
        super(SuGCN, self).__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.linear  = nn.Linear(self.encoder.nhid, num_classes)
    def forward(self, feat, adjs):
        x = self.dropout(self.encoder(feat, adjs))
        x = self.linear(x)
        return F.log_softmax(x, dim=1)


# In[ ]:


def fastgcn_sampler(seed, batch_nodes, samp_num_list, num_nodes, lap_matrix, depth):
    np.random.seed(seed)
    previous_nodes = batch_nodes
    adjs  = []
    pi = np.array(np.sum(lap_matrix.multiply(lap_matrix), axis=0))[0]
    p = pi / np.sum(pi)
    for d in range(depth):
        U = lap_matrix[previous_nodes , :]
        s_num = np.min([np.sum(p > 0), samp_num_list[d]])
        after_nodes = np.random.choice(num_nodes, s_num, p = p, replace = False)
        adj = row_norm(U[: , after_nodes].multiply(1/p[after_nodes]))
        sparse_mx = adj.tocoo().astype(np.float32)
        adjs += [sparse_mx_to_torch_sparse_tensor(row_norm(adj))]
        previous_nodes = after_nodes
    adjs.reverse()
    return adjs, previous_nodes, batch_nodes

def ladies_sampler(seed, batch_nodes, samp_num_list, num_nodes, lap_matrix, depth):
    np.random.seed(seed)
    previous_nodes = batch_nodes
    adjs  = []
    for d in range(depth):
        U = lap_matrix[previous_nodes , :]
        pi = np.array(np.sum(U.multiply(U), axis=0))[0]
        p = pi / np.sum(pi)
        s_num = np.min([np.sum(p > 0), samp_num_list[d]])
        after_nodes = np.random.choice(num_nodes, s_num, p = p, replace = False)
        adj = U[: , after_nodes].multiply(1/p[after_nodes])
        adjs += [sparse_mx_to_torch_sparse_tensor(row_norm(adj))]
        previous_nodes = after_nodes
    adjs.reverse()
    return adjs, previous_nodes, batch_nodes
def default_sampler(seed, batch_nodes, samp_num_list, num_nodes, lap_matrix, depth):
    mx = sparse_mx_to_torch_sparse_tensor(lap_matrix)
    return [mx for i in range(depth)], np.arange(num_nodes), batch_nodes
def prepare_data(pool, process_ids, train_nodes, samp_num_list, num_nodes, lap_matrix, depth):
    jobs = []
    for _ in process_ids:
        idx = torch.randperm(len(train_nodes))[:args.batch_size]
        batch_nodes = train_nodes[idx]
        p = pool.apply_async(ladies_sampler, args=(np.random.randint(2**32 - 1), batch_nodes,                                                    samp_num_list, num_nodes, lap_matrix, depth))
        jobs.append(p)
    return jobs
def package_mxl(mxl, device):
    return [torch.sparse.FloatTensor(mx[0], mx[1], mx[2]).to(device) for mx in mxl]


# In[ ]:


if args.cuda != -1:
    device = torch.device("cuda:" + args.cuda)
else:
    device = torch.device("cpu")
edges, labels, feat_data, num_classes, train_nodes, valid_nodes, test_nodes = load_data(args.dataset)


# In[ ]:


adj_matrix = get_adj(edges, feat_data.shape[0])
adj_matrix = adj_matrix + sp.eye(adj_matrix.shape[0])
lap_matrix = normalize(adj_matrix)

if type(feat_data) == scipy.sparse.lil.lil_matrix:
    feat_data = torch.FloatTensor(feat_data.todense()).to(device) 
else:
    feat_data = torch.FloatTensor(feat_data).to(device)
labels    = torch.LongTensor(labels).to(device) 


# In[ ]:


if args.sample_method == 'ladies':
    sampler = ladies_sampler
elif args.sample_method == 'fastgcn':
    sampler = fastgcn_sampler
elif args.sample_method == 'full':
    sampler = default_sample


# In[ ]:


process_ids = np.arange(args.batch_num)
samp_num_list = [64 for i in range(args.n_layers)]

pool = mp.Pool(args.pool_num)
jobs = prepare_data(pool, process_ids, train_nodes, samp_num_list, len(feat_data), lap_matrix, args.n_layers)

std_adjs, _, _ = default_sampler(_, _, _, 0, lap_matrix, args.n_layers)
std_adjs = package_mxl(std_adjs, device)
all_res = []
for o_iter in range(5):
    encoder = GCN(nfeat = feat_data.shape[1], nhid=args.nhid, layers=args.n_layers, dropout = 0.2).to(device)
    susage  = SuGCN(encoder = encoder, num_classes=num_classes, dropout=0.5, inp = feat_data.shape[1])
    susage.to(device)

    optimizer = optim.Adam(filter(lambda p : p.requires_grad, susage.parameters()))
    best_val = -1
    best_tst = -1
    cnt = 0
    times = []
    res   = []
    for epoch in np.arange(args.epoch_num):
        susage.train()
        train_data = [job.get() for job in jobs]
        pool.close()
        pool.join()
        pool = mp.Pool(args.pool_num)
        jobs = prepare_data(pool, process_ids, train_nodes, samp_num_list, len(feat_data), lap_matrix, args.n_layers)
        for adjs, input_nodes, output_nodes in train_data:    
            adjs = package_mxl(adjs, device)
            optimizer.zero_grad()
            t1 = time.time()
            susage.train()
            output = susage.forward(feat_data[input_nodes], adjs)
            loss_train = F.cross_entropy(output, labels[output_nodes])
            loss_train.backward()
            optimizer.step()
            times += [time.time() - t1]
            del loss_train
        if epoch % 1 == 0:
            susage.eval()
            pred = susage.forward(feat_data, std_adjs)
            
            trai_label = labels[train_nodes]
            trai_pred = pred[train_nodes].cpu().detach().numpy()
            trai_f1 = f1_score(np.argmax(trai_pred, axis=1), trai_label.cpu().detach().numpy(), average='micro')    

            vald_label = labels[valid_nodes]
            vald_pred = pred[valid_nodes].cpu().detach().numpy()
            vald_f1 = f1_score(np.argmax(vald_pred, axis=1), vald_label.cpu().detach().numpy(), average='micro')
            
            test_label = labels[test_nodes]
            test_pred = pred[test_nodes].cpu().detach().numpy()
            test_f1 = f1_score(np.argmax(test_pred, axis=1), test_label.cpu().detach().numpy(), average='micro')
            
            res += [[trai_f1, vald_f1, test_f1]]
            if vald_f1 >= best_val + 1e-2:
                best_val = vald_f1
                best_tst = test_f1
                cnt = 0
            else:
                cnt += 1
            if cnt == 20:
                break
                
    res = np.array(res)
    plt.plot(res[:,0], label='train')
    plt.plot(res[:,1], label='valid')
    plt.plot(res[:,2], label='test')
    best_batch = np.argmax(res[:,1])
    plt.title("Test F1: %.3f, Time: %.3f, Batch Num: %.3f" % (res[:,2][best_batch], np.sum(times[:best_batch * args.batch_num]), best_batch * args.batch_num))
    plt.legend()
    plt.show()
    for epoch, (trai_f1, vald_f1, test_f1) in enumerate(res):
        all_res += [[trai_f1, epoch * args.batch_num, 'train']]
        all_res += [[vald_f1, epoch * args.batch_num, 'valid']]
        all_res += [[test_f1, epoch * args.batch_num, 'test']]


# In[ ]:


dt = pd.DataFrame(all_res, columns=['f1-score', 'batch', 'type'])
sb.lineplot(data = dt, x='batch', y='f1-score', hue='type')
plt.legend(loc='lower right')
plt.show()

