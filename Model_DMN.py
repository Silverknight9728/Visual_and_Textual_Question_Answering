import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from copy import deepcopy
def flatten(l): return [item for sublist in l for item in sublist]

random.seed(1024)

FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor
ByteTensor = torch.ByteTensor

MSS = 0   #MAX_SENT_SIZE = MSS
HS = 80 #HIDDEN_SIZE = HS
BS = 32 # BATCH_SIZE = BS
LR = 0.001 
EPOCH = 5 
NUM_EPS = 3 #NUM_EPISODE = NUM_EPS
EP = False #EARLY_STOPPING = EP



def getBatch(batch_size, train_data):
    random.shuffle(train_data)
    start_index = 0
    end_index = batch_size
    while end_index < len(train_data):
        temp = end_index
        batch = train_data[start_index: end_index]
        end_index += batch_size
        start_index = temp
        yield batch

    if end_index >= len(train_data):
        yield train_data[start_index:]


def padToBatch(batch, w_ix): #pad_to_batch = padToBatch

    global MSS

    fact, ques, ans = list(zip(*batch))  #fact = fact, q = ques, and a = ans

    # fetch the max number of facts for a sample 
    longest_fact = max([len(i) for i in fact]) #max_fact = longest_fact
    maximum_len = max([i.size(1) for i in flatten(fact)]) #max_len = maximum_len
    MSS = maximum_len

    maximum_ques = max([k.size(1) for k in ques]) #max_q = maximum_ques
    maximum_ans = max([s.size(1) for s in ans]) #max_a = maximum_ans

    facts = []
    ques_p = []
    ans_p = []
    factMasks = []
    #fact_mask = factMasks , q_p = ques_p, a_p= ans_p
    for i in range(len(batch)):
        factpt = [] #fact_p_t = factpt
        for j in range(len(fact[i])):
            if fact[i][j].size(1) < maximum_len:
                # append each fact with id of '' to max fact length
                factpt.append(torch.cat([fact[i][j], Variable(LongTensor(
                    [w_ix['<PAD>']] * (maximum_len - fact[i][j].size(1)))).view(1, -1)], 1))
            else:
                factpt.append(fact[i][j])

        # append empty facts to match largest number of facts
        while len(factpt) < longest_fact:
            factpt.append(
                Variable(LongTensor([w_ix['<PAD>']] * maximum_len)).view(1, -1))

        factpt = torch.cat(factpt)
        facts.append(factpt)

        # wherever value is 0 make it 1
        factMasks.append(torch.cat([Variable(ByteTensor(tuple(map(
            lambda s: s == 0, t.data))), volatile=False) for t in factpt]).view(factpt.size(0), -1))

        if ans[i].size(1) < maximum_ans:
            ans_p.append(torch.cat([ans[i], Variable(LongTensor(
                [w_ix['<PAD>']] * (maximum_ans - ans[i].size(1)))).view(1, -1)], 1))
        else:
            ans_p.append(ans[i])
        
        if ques[i].size(1) < maximum_ques:
            ques_p.append(torch.cat([ques[i], Variable(LongTensor(
                [w_ix['<PAD>']] * (maximum_ques - ques[i].size(1)))).view(1, -1)], 1))
        else:
            ques_p.append(ques[i])

    questions = torch.cat(ques_p)
    answers = torch.cat(ans_p)
    questionMasks = torch.cat([Variable(ByteTensor(tuple(map(
        lambda s: s == 0, t.data))), volatile=False) for t in questions]).view(questions.size(0), -1)
    #question_masks = questionMasks
    return facts, factMasks, questions, questionMasks, answers


def sequencePrep(seq, to_index): #prepare_sequence = sequencePrep
    # Use the word to index dict to generate a sequence of numbers
    idxs = list(map(lambda w: to_index[w] if to_index.get(
        w) is not None else to_index["<UNK>"], seq))
    return Variable(LongTensor(idxs))


file = open(
    'data/tasks_1-20_v1-2/en/qa5_three-arg-relations_train.txt').readlines()
# Removes \n character
d = [l[:-1] for l in file] #data_file = file z data = d
train_data = []
fact = []
for line in d:
    idx = line.split(' ')[0] #index = idx
    if(idx == '1'):
        # Reset fact
        fact = []
    if('?' in line):
        temp = line.split('\t')

        # Question and ans both are split into tokens
        question = temp[0].strip().replace('?', '').split(' ')[1:] + ['?']
        answer = temp[1].split() + ['</s>']
        listOfFacts = deepcopy(fact) #list_of_facts = listOfFacts

        # Finally when we have the question answer
        # temp_s 2D, ques 1D, ans 1D
        train_data.append([listOfFacts, question, answer])
    else:
        # remove "." and the index number of the fact
        # Also append an empty space at the end of the list (idk why yet)
        # each fact is a set of tokens
        fact.append(line.replace('.', '').split(' ')[1:] + ['</s>'])

fact, ques, ans = list(zip(*train_data)) #q = ques, a = ans
# Build a vocab of all the words in our facts, ques and ans
v_fqa = list(set(flatten(flatten(fact)) + flatten(ques) + flatten(ans))) #vocabulary =v_fqa


wordToIndex = {'<PAD>': 0, '<UNK>': 1, '<s>': 2, '</s>': 3} #word_to_index = wordToIndex
# word_to_index = dict()
for word in v_fqa:
    if wordToIndex.get(word) is None:
        wordToIndex[word] = len(wordToIndex)
indexToWord = {v: k for k, v in wordToIndex.items()} #index_to_word = indexToWord


for k in train_data:
    for i, fact in enumerate(k[0]):
        k[0][i] = sequencePrep(fact, wordToIndex).view(1, -1)
    k[1] = sequencePrep(k[1], wordToIndex).view(1, -1)
    k[2] = sequencePrep(k[2], wordToIndex).view(1, -1)


class AttnGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AttnGRU, self).__init__()

        self.hidden_size = hidden_size
        #self.num_layers = num_layers

        # Weight matrices for reset gate
        self.Wr = nn.Linear(hidden_size, hidden_size)
        self.Ur = nn.Linear(hidden_size, hidden_size)

        # Weight matrices for update gate
        self.Wz = nn.Linear(hidden_size, hidden_size)
        self.Uz = nn.Linear(hidden_size, hidden_size)

        # Weight matrices for candidate activation
        self.W = nn.Linear(hidden_size, hidden_size)
        self.U = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, h0, gt):
        # Initialize hidden state tensor
        h = h0
        # Loop through time steps
        for t in range(x.size(1)):
            # Concatenate input and previous hidden state
            xt = x[:, t, :]
            ht = h

#             xh = torch.cat([xt, ht], dim=1)
            xh = xt
#             print("xh shape", xh.shape)
            # Compute reset gate
            rt = torch.sigmoid(self.Wr(xh) + self.Ur(ht))

            # Compute update gate
            zt = torch.sigmoid(self.Wz(xh) + self.Uz(ht))

            # Compute candidate activation
            nt = torch.tanh(self.W(xh) + rt * self.U(ht))

            # Update hidden state
            h = (1 - gt) * nt + gt * ht

        # Return output and final hidden state tensor
        return h


class DMN(nn.Module):
    def __init__(self, size_of_input, size_of_hidden, size_of_output , d_p=0.1):
        #input_size = size_of_hidden
        #hidden_size = size_of_hidden
        #output_size = size_of_hidden 
        #dropout_p = d_p

        super(DMN, self).__init__()

        self.size_of_hidden = size_of_hidden

        self.embedding = nn.Embedding(size_of_hidden, size_of_hidden)
        self.fact_gru = nn.GRU(size_of_hidden, size_of_hidden, batch_first=True)
        self.ques_gru = nn.GRU(size_of_hidden, size_of_hidden, batch_first=True)
        self.attn_weights = nn.Sequential(nn.Linear(
            4*size_of_hidden, size_of_hidden), nn.Tanh(), nn.Linear(size_of_hidden, 1), nn.Softmax())


        self.epsisodic_grucell = AttnGRU(size_of_hidden, size_of_hidden)
        self.memory_grucell = nn.GRUCell(size_of_hidden, size_of_hidden)
        self.memory_linear = nn.Sequential(
            nn.Linear(3*size_of_hidden, size_of_hidden), nn.ReLU())

        self.ans_grucell = nn.GRUCell(2*size_of_hidden, size_of_hidden)

        self.ans_fc = nn.Linear(size_of_hidden, size_of_hidden)

        self.dropout = nn.Dropout(d_p)

    def encodingPositional(self): #positional_encoding = encodingPositional 
        soh, M = self.size_of_hidden, MSS #D = soh
        encoding = torch.zeros([M, soh])
        for j in range(M):
            for d in range(soh):
                encoding[j, d] = (1 - float(j)/M) - (float(d)/soh)*(1 - 2.0*j/M)

        return encoding

    def init_hidden(self, inputs):
        hidden = Variable(torch.zeros(1, inputs.size(0), self.size_of_hidden))
        return hidden

    def init_weight(self):
        nn.init.xavier_uniform_(self.embedding.state_dict()['weight'])

        for name, param in self.fact_gru.state_dict().items():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
        for name, param in self.ques_gru.state_dict().items():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
        for name, param in self.attn_weights.state_dict().items():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
        for name, param in self.epsisodic_grucell.state_dict().items():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
        for name, param in self.memory_grucell.state_dict().items():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
        for name, param in self.ans_grucell.state_dict().items():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

        nn.init.xavier_normal_(self.ans_fc.state_dict()['weight'])
        self.ans_fc.bias.data.fill_(0)

    def forward(self, facts, facts_masks, question, question_masks, num_decode, episodes=3, is_training=True):
        # input module
        c_fw = [] #c_fw
        c_bw = [] #c_bw
        for fact in facts:

            # Convert every f (word indexes) in fact to a tensor of len(f) * hidden dim
            # essetially every word has an 80 dim embedding vector
            embedded = self.embedding(fact)
            if(is_training):
                # randomly set some of the elements to 0
                embedded = self.dropout(embedded)
            # get a hidden matrix of zeros of shape 1, batch size, hidden dim
            hidden = self.init_hidden(fact)
            # the gru layer takes embedding and h_0 as the input. gives output and new hidden state
            output, hidden = self.fact_gru(embedded, hidden)
            # flipped output for backward pass
            hidden = self.init_hidden(fact)
            output_bw, hidden_bw = self.fact_gru(
                torch.flip(embedded, dims=[1]), hidden)
            hidden_state_fw = []
            for i, o in enumerate(output):
                # 0s indicate useful information and theyre only on the left side
                #                 length = fact_mask[i].data.tolist().count(0)
                #                 hidden_real.append(o[length-1])

                # we append the last useful info position in hidden_real (idk why yet)
                hidden_state_fw.append(sum(self.positional_encoding()*o))
            c_fw.append(torch.cat(hidden_state_fw).view(
                fact.size(0), -1).unsqueeze(0))

            hidden_state_bw = []
            for i, o in enumerate(output_bw):

                # we append the last useful info position in hidden_real (idk why yet)
                hidden_state_bw.append(sum(self.positional_encoding()*o))
            c_bw.append(torch.cat(hidden_state_bw).view(
                fact.size(0), -1).unsqueeze(0))

        factsEncoded = torch.cat(c_fw) + \
            torch.flip(torch.cat(c_bw), dims=[1])
        #encoded_facts = factsEncoded

        # question module
        hidden = self.init_hidden(question)

        embedded = self.embedding(question)
        if(is_training):
            embedded = self.dropout(embedded)
        output, hidden = self.ques_gru(embedded, hidden)

        if is_training == True:
            real_question = []
            for i, o in enumerate(output):  # B,T,D
                real_length = question_masks[i].data.tolist().count(0)

                real_question.append(o[real_length - 1])

            encoded_question = torch.cat(real_question).view(
                questions.size(0), -1)  # B,D
        else:  # for inference mode
            encoded_question = hidden.squeeze(0)  # B,D

        # episodic memory module
        # initialize memory with q
        memory = encoded_question
        T_C = factsEncoded.size(1)  # max fact count
#         print(T_C)
        B = factsEncoded.size(0)  # batch size
        for i in range(episodes):
            hidden = self.init_hidden(
                factsEncoded.transpose(0, 1)[0]).squeeze(0)  # B,D
            for t in range(T_C):

                z = torch.cat([
                    # B,D , element-wise product
                    factsEncoded.transpose(0, 1)[t] * encoded_question,
                    # B,D , element-wise product
                    factsEncoded.transpose(0, 1)[t] * memory,
                    torch.abs(factsEncoded.transpose(0, 1)[
                              t] - encoded_question),  # B,D
                    torch.abs(factsEncoded.transpose(0, 1)[t] - memory)  # B,D
                ], 1)
                g_t = self.attn_weights(z)  # B,1 scalar

                hidden = self.epsisodic_grucell(factsEncoded, hidden, g_t)

            e = hidden
            prev_mem = memory

            memory = self.memory_grucell(e, memory)
            concatenated_input = torch.cat(
                [prev_mem, e, encoded_question], dim=1)
            memory = self.memory_linear(concatenated_input)

        # Answer Module
        answer_hidden = memory
        start_decode = Variable(LongTensor(
            [[wordToIndex['<s>']] * memory.size(0)])).transpose(0, 1)
        y_t_1 = self.embedding(start_decode).squeeze(1)  # B,D

        decodes = []
        for t in range(num_decode):
            answer_hidden = self.ans_grucell(
                torch.cat([y_t_1, encoded_question], 1), answer_hidden)
            decodes.append(F.log_softmax(self.ans_fc(answer_hidden), 1))
        return torch.cat(decodes, 1).view(B * num_decode, -1)


model = DMN(len(wordToIndex), HS, len(wordToIndex))
model.init_weight()

loss_function = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCH):
    losses = []
    if EARLY_STOPPING:
        break

    for i, batch in enumerate(getBatch(BS, train_data)):
        facts, fact_masks, questions, question_masks, answers = padToBatch(
            batch, wordToIndex)

        model.zero_grad()
        pred = model(facts, fact_masks, questions, question_masks,
                     answers.size(1), NUM_EPS, True)
        loss = loss_function(pred, answers.view(-1))
        losses.append(float(loss.data))

        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print("[%d/%d] mean_loss : %0.2f" %
                  (epoch, EPOCH, np.mean(losses)))

            if np.mean(losses) < 0.01:
                EARLY_STOPPING = True
                print("Early Stopping!")
                break
            losses = []

torch.save(model, 'DMN_plus.pickle')


def padToFact(fact, x_to_ix):  # this is for inference 
    #pad_to_fact = padToFact

    maximum_x = max([s.size(1) for s in fact]) #max_x = maximum_x
    x_p = []
    for i in range(len(fact)):
        if fact[i].size(1) < maximum_x:
            x_p.append(torch.cat([fact[i], Variable(LongTensor(
                [x_to_ix['<PAD>']] * (maximum_x - fact[i].size(1)))).view(1, -1)], 1))
        else:
            x_p.append(fact[i])

    fact = torch.cat(x_p)
    factMask = torch.cat([Variable(ByteTensor(tuple(map(
        lambda s: s == 0, t.data))), volatile=False) for t in fact]).view(fact.size(0), -1)
    return fact, factMask

#fact_mask = factMask
data = open(
    'data/tasks_1-20_v1-2/en/qa5_three-arg-relations_test.txt').readlines()
data = [d[:-1] for d in data]
test_data = []
fact = []
for d in data:
    index = d.split(' ')[0]
    if(index == '1'):
        fact = []
    if('?' in d):
        temp = d.split('\t')
        ques = temp[0].strip().replace('?', '').split(' ')[1:] + ['?']
        ans = temp[1].split() + ['</s>']
        temp_s = deepcopy(fact)
        test_data.append([temp_s, ques, ans])
    else:
        fact.append(d.replace('.', '').split(' ')[1:] + ['</s>'])

for t in test_data:
    for i, fact in enumerate(t[0]):
        t[0][i] = sequencePrep(fact, wordToIndex).view(1, -1)

    t[1] = sequencePrep(t[1], wordToIndex).view(1, -1)
    t[2] = sequencePrep(t[2], wordToIndex).view(1, -1)

accuracy = 0
for t in test_data:
    fact, factMask = padToFact(t[0], wordToIndex)
    question = t[1]
    questionMask = Variable(ByteTensor(
        [0] * t[1].size(1)), requires_grad=False).unsqueeze(0)
    answer = t[2].squeeze(0)

    model.zero_grad()
    pred = model([fact], [factMask], question, questionMask,
                 answer.size(0), NUM_EPS, False)
    if pred.max(1)[1].data.tolist() == answer.data.tolist():
        accuracy += 1

print(accuracy/len(test_data) * 100)


t = random.choice(test_data)
fact, factMask = padToFact(t[0], wordToIndex)
question = t[1]
questionMask = Variable(ByteTensor(
    [0] * t[1].size(1)), requires_grad=False).unsqueeze(0)
answer = t[2].squeeze(0)

model.zero_grad()
pred = model([fact], [factMask], question, questionMask,
             answer.size(0), NUM_EPS, False)

print("Facts : ")
print('\n'.join([' '.join(list(map(lambda x: indexToWord[x], f)))
      for f in fact.data.tolist()]))
print("")
print("Question : ", ' '.join(
    list(map(lambda x: indexToWord[x], question.data.tolist()[0]))))
print("")
print("Answer : ", ' '.join(
    list(map(lambda x: indexToWord[x], answer.data.tolist()))))
print("Prediction : ", ' '.join(
    list(map(lambda x: indexToWord[x], pred.max(1)[1].data.tolist()))))
