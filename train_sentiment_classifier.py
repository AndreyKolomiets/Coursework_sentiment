# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
use_gpu = torch.cuda.is_available()
if use_gpu:
    print('cuda is availiable')
else:
    print('cuda not availiable')
import gensim

from tqdm import tqdm_notebook, trange

from sklearn.model_selection import train_test_split
from datetime import datetime

from IPython.display import clear_output


# !sudo pip install gensim

# In[2]:


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    if shuffle: 
        indices = np.random.permutation(len(inputs))
    for start_idx in trange(0, len(inputs) + 1, batchsize):
        #print(start_idx)
        #print(start_idx + batchsize)
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


# Протестируем iterate_minibatches. Готовая функция из семинаров немного косячила (не выдавала последнюю часть размера input_len % batch_size)

# In[3]:


XX = np.random.normal(size=(111, 5))
YY = np.random.normal(size=111)
for x_batch, y_batch in iterate_minibatches(XX, YY, 95):
    print(x_batch.shape, y_batch.shape)


# In[4]:


data = pd.read_csv('/data/Coursework/all_data_normalized.csv')
data.head()



root_path = '/data/Coursework/'
embedding_model = gensim.models.Word2Vec
embedding_model = embedding_model.load(root_path + 'model_normalized_with_chats_2.bin')


# Уберем строки, в которых после нормализации нет ни одного слова

# In[7]:


data = data[data['len'] > 0]


# In[8]:


data.score.value_counts(dropna=False)


# Уберем из данных тексты с промежуточными оценками. Можно было бы и распределить по оставшимся, но это неоднозначное решение

# In[9]:


data = data[data.score.isin([1, 3, 5])]


# In[10]:


data.score.value_counts()


# In[11]:


data.dropna(inplace=True, subset=['text_normalized'])


# Для удобства работы сортируем датафрейм

# In[12]:


data.sort_values(by=['len'], ascending=False, inplace=True)


# Заменим метки классов на 0,1,2. Иначе нужно танцевать с бубном, чтобы Pytorch правильно нашел метки и количество классов

# In[13]:


mapping = {1:0, 3:1, 5:2}
data.score = data.score.replace(mapping)


# In[14]:


data.head()


# Собственно модель

# In[58]:


class SentimentRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers, cell='rnn'):
        super(SentimentRNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.cell = cell
        cell = cell.lower()
        if cell == "rnn":
            self.rnn = nn.RNN(input_size, hidden_size, n_layers)
        elif cell == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_size, n_layers)
        elif cell == 'gru':
            self.rnn = nn.GRU(input_size, hidden_size, n_layers)
        else:
            raise ValueError('Тип ячейки должен быть lstm, gru или rnn')
        self.decoder = nn.Linear(hidden_size, output_size)
        self.output_size = output_size
        self.grad_ih = None
        self.grad_hh = None
        
    def forward(self, input, hidden):
        # print(type(input.data), type(hidden.data))
        #print('input shape:', input.shape, '  hidden shape:', hidden.shape)
        print(type(input.data), type(hidden.data))
        output, hidden = self.rnn(input, hidden)
        #res = self.decoder(output)
        #return res.view(-1, self.output_size), hidden
        return output, hidden
    
    def init_hidden(self, batch_size):
        hidden = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))
        if torch.cuda.is_available():
            hidden = hidden.cuda()
        if self.cell == "lstm":
            c0 = Variable(torch.randn(self.n_layers, batch_size, self.hidden_size))
            if torch.cuda.is_available():
                c0 = c0.cuda()
            return (hidden, c0)
        return hidden
    
    # Метод для поиска l2-норм градиентов
    # Веса каждого слоя хранятся в своём атрибуте, для извлечения в цикле нужен getattr
    def get_grad_norms(self):
        grads_ih = np.zeros(self.n_layers)
        grads_hh = np.zeros(self.n_layers)
        for i in range(self.n_layers):
            weight_ih = getattr(self.rnn, 'weight_ih_l'+str(i))
            ih_norm = torch.norm(weight_ih.grad.view(-1, 1), p=2, dim=0).data.numpy()[0]
            grads_ih[i] = ih_norm
            
            weight_hh = getattr(self.rnn, 'weight_hh_l'+str(i))
            grads_hh[i] = torch.norm(weight_hh.grad.view(-1, 1), p=2, dim=0).data.numpy()[0]
        return grads_ih, grads_hh
    
    def predict_proba(self, x, hidden):
        out, _ = self.forward(x, hidden)
        probas = F.softmax(out, dim=1)
        return probas


# Функция, преобразующая список предложений в вектора word2vec 

# In[59]:


def embed(sentences, embedder):
    l = len(sentences[0].split())
    n = embedder.wv.vector_size
    bs = len(sentences)
    res = np.zeros([l, bs, n])
    # Вектор для неизвестных классификатору слов
    unknown = np.ones(embedding_size).astype('float32') * 7
    for i in range(bs):
        words = sentences[i].split()
        for j in range(l):
            try:
                res[j, i, :] = embedder.wv[words[j]]
            except KeyError:
                res[j, i, :] = unknown
    return res


# Поскольку разобраться с pack_padded_sequences пока не удалось, то будем подавать батчи из предложений одинаковой длины. Значения длины перебираем в случайном порядке (иначе мб циклическое изменение функции потерь вместо уменьшения)

# In[60]:


def train_epoch(model, optimizer, inputs, targets):
    loss_log = []
    model.train()
    #print('batch_size: ', batch_size)
    for X, y in iterate_minibatches(inputs, targets, batch_size, shuffle=True):
        t_start = datetime.now()
        x_batch = embed(X, embedding_model)
        #print(np.unique(y))
        #print('Time to embed:', datetime.now() - t_start)
        t_start = datetime.now()
        print('batch')
        if use_gpu:
            x_batch = Variable(torch.from_numpy(x_batch).cuda())
            y_batch = Variable(torch.from_numpy(y).cuda())
        else:
            x_batch = Variable(torch.from_numpy(x_batch).type(torch.FloatTensor))
            y_batch = Variable(torch.from_numpy(y))
        print(type(x_batch))
        #print('Time to make variables:', datetime.now() - t_start)
        #print('batch shape:', x_batch.shape)
        hidden = model.init_hidden(x_batch.size(1))
        #print('hidden shape:', hidden.shape, '  x shape:', x_batch.shape, '  y shape:', y_batch.shape)
        optimizer.zero_grad()
        loss = 0.0
        t_start = datetime.now()
        for i in range(x_batch.size(0)):
            print('seq')
            out, hidden = model.forward(x_batch[i, ...].view(1, x_batch.size(1), x_batch.size(2)), hidden)
            #print(out.shape, hidden.shape)
        res = model.decoder(out)
        
        loss = F.cross_entropy(res.view(-1, model.output_size), y_batch)
        #out, hidden = model.forward(x_batch, hidden)
        
        #print(out.shape, hidden.shape, y_batch.shape)
        # loss += F.cross_entropy(out, y_batch)
        #print('Time to forward propagation:', datetime.now() - t_start)
        t_start = datetime.now()
        loss.backward()
        #print('Time to backprop:', datetime.now() - t_start)
        t_start = datetime.now()
        optimizer.step()
        #print('Time to optimizer step:', datetime.now() - t_start)
        loss = loss.data[0]
        loss_log.append(loss)
    return loss_log   

def test(model, inputs, targets):
    loss_log = []
    model.eval()
    for X, y in iterate_minibatches(inputs, targets, batch_size):
        x_batch = embed(X, embedding_model)
        if use_gpu:
            x_batch = Variable(torch.from_numpy(x_batch).cuda(0))
            y_batch = Variable(torch.from_numpy(y).cuda(0))
        else:
            x_batch = Variable(torch.from_numpy(x_batch).type(torch.FloatTensor))
            y_batch = Variable(torch.from_numpy(y))
        hidden = model.init_hidden(x_batch.size(1))
        loss = 0.0
        for i in range(x_batch.size(0)):
            out, hidden = model.forward(x_batch[i, ...].view(1, x_batch.size(1), x_batch.size(2)), hidden)
        res = model.decoder(out)
        loss = F.cross_entropy(res.view(-1, model.output_size), y_batch)
        #out, hidden = model.forward(x_batch, hidden)
        #loss += F.cross_entropy(out, y_batch)
        loss = loss.data[0]
        loss_log.append(loss)
    return loss_log

def plot_history(train_history, val_history, title='loss'):
    plt.figure()
    plt.title('{}'.format(title))
    plt.plot(train_history, label='train', zorder=1)    
    points = np.array(val_history)

    plt.scatter(points[:, 0], points[:, 1], marker='+', s=180, c='orange', label='val', zorder=2)

    plt.xlabel('train steps')
    plt.legend(loc='best')
    plt.grid()
    plt.show()
    
def train(model, opt, n_epochs):
    unique_lengths = np.unique([min(x, max_length) for x in data['len'].unique()])
    #lengths = data['len'].values
    # Индексы, в которых появляются новые длины
    #change_indices = np.where(np.diff(lengths) < 0)[0] + 1
    
    train_log = []
    val_log = []
    steps = 0
    # Сюда будем писать нормы градиентов
    grads_ih = np.zeros([len(unique_lengths) * n_epochs, model.n_layers])
    #print(grads_ih.shape)
    grads_hh = np.zeros([len(unique_lengths) * n_epochs, model.n_layers])
    for epoch in range(n_epochs):
        for i, ul in enumerate(np.random.permutation(unique_lengths)):
            data_cur = data[data['len'] == ul]
            print('Length:', ul)
            X = data_cur.text_normalized.values
            y = data_cur.score.values
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=234769)
            #print('data shape:', X_train.shape, X_test.shape)
            steps += X_train.shape[0] / batch_size
            train_loss = train_epoch(model, opt, X_train, y_train)
            train_log.extend(train_loss)

            val_loss = test(model, X_test, y_test)
            val_log.append((steps, np.mean(val_loss)))
            clear_output()
            plot_history(train_log, val_log)
            #Собственно получение градиентов
            grad_ih, grad_hh = model.get_grad_norms()
            grads_ih[i * (epoch + 1), :] = grad_ih
            grads_hh[i * (epoch + 1), :] = grad_hh
    model.grad_ih = grads_ih
    model.grad_hh = grads_hh
    return train_log, val_log


# Выбираем максимальную длину предложения. Критерий - должны быть учтены 99% предложений

# In[61]:


max_length = int(np.percentile(data['len'], 99))
max_length


# In[62]:


torch.cuda.device_count()


# In[67]:


torch.backend.cudnn.enabled=False


# In[64]:


get_ipython().run_cell_magic(u'time', u'', u"n_layers = 1\nbatch_size = 1024\nhidden_size = 250\noutput_size = 3\nembedding_size = embedding_model.wv.vector_size\nn_epochs = 1\nver = 5\n\nmodel = SentimentRNN(embedding_size, hidden_size, output_size, n_layers, cell='rnn')\n#sd = torch.load('sentiment_clf_4.pth')\n#model.load_state_dict(sd)\nif use_gpu:\n    model = model.cuda()\n\nopt = torch.optim.Adam(model.parameters(), lr=1e-4)\ntrain_log, val_log = train(model, opt, n_epochs)\ntorch.save(model.state_dict(), 'sentiment_clf_%i.pth'%ver)")


# In[65]:


torch.backends.cudnn.version()


# In[66]:


print(torch.backends.cudnn.is_acceptable(torch.cuda.FloatTensor(1)))
print(torch.backends.cudnn.version())


# In[ ]:


get_ipython().system(u'top')


# In[ ]:


get_ipython().system(u'nvcc --version')


# In[ ]:


val_log


# In[ ]:


data['len'].unique()


# In[ ]:


21375 % batch_size


# Отрисовка градиентов

# In[ ]:



legend_g = [str(i + 1) for i in range(model.n_layers)]
# absc = np.array(range(1, n_epochs + 1)) * steps
steps = 0.67 * len(data) / batch_size
absc = np.array([xx[0] for xx in val_log]) 

plt.figure()
plt.grid()
plt.title('Gradient norm input to hidden vs train steps')
for i in range(model.n_layers):
    plt.plot(absc, model.grad_ih[:, i])
plt.legend(legend_g)
plt.xlabel('train steps')
plt.gca().set_yscale('log')

plt.figure()
plt.grid()
plt.title('Gradient norm hidden to hidden vs train steps')
for i in range(model.n_layers):
    plt.plot(absc, model.grad_hh[:, i])
plt.legend(legend_g)
plt.xlabel('train steps')
plt.gca().set_yscale('log')


absc = np.arange(0, model.n_layers, 1) + 1
legend_g = [str(i + 1) for i in range(n_epochs)]
plt.figure()
plt.grid()
plt.title('Gradient norm input to hidden vs layer num')
for i in range(n_epochs):
    plt.plot(absc, model.grad_ih[i, :])
plt.legend(legend_g)
plt.xlabel('layer num')
plt.gca().set_yscale('log')

plt.figure()
plt.grid()
plt.title('Gradient norm hidden to hidden vs layer num')
for i in range(n_epochs):
    plt.plot(absc, model.grad_hh[i, :])
plt.legend(legend_g)
plt.xlabel('layer num')
plt.gca().set_yscale('log')


# In[ ]:


model.grad_ih.shape


# In[31]:


get_ipython().system(u'nvidia-smi')


# In[30]:


sd.keys()

