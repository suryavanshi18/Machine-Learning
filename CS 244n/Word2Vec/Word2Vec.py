
# %%
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import random
from collections import deque
import matplotlib.pyplot as plt
import json
# %%
nltk.download('punkt')# Split sentence into words
nltk.download('stopwords')
nltk.download('punkt_tab')
file_path='small_corpus.txt'
with open(file_path,'r',encoding='utf-8') as file:
    text=file.read()
# %%
def extract_unique_words(text):
    text=text.lower()
    words=word_tokenize(text)

    #Remove punctuations
    words=[word for word in words if word.isalpha()]
    print("No. of words in text:",len(words))
    unique_words=sorted(set(words))
    print("No of unique words",len(unique_words))
    return unique_words
unique_words = extract_unique_words(text)
# %%
def generate_cbows(text,windows_size):
    text=text.lower()
    words=word_tokenize(text)
    # Remove punctuation
    words = [word for word in words if word.isalpha()]
    #Remove stop words
    stop_words=set(stopwords.words('english'))
    words=[word for word in words if word not in stop_words]
    cbows=[]
    for i,target_word in enumerate(words):
        context_words=words[max(0,i-windows_size):i]+words[i+1:i+windows_size+1]
        if len(context_words)==windows_size*2:
            cbows.append((context_words,target_word))
    return cbows
cbows=generate_cbows(text,windows_size=3)
for context,target in cbows[:2]:
    print(context," target_word: ",target)
len(cbows)
# %%
def one_hot_encoding(word,unique_words):
    encoding=[1 if word==w else 0 for w in unique_words]
    return torch.tensor(encoding,dtype=torch.float32)
one_hot_encodings={word:one_hot_encoding(word,unique_words) for word in unique_words}

# %%
#Convert cbow pairs to context vectors
cbow_vector_pairs=[([one_hot_encodings[word] for word in context_word],one_hot_encodings[target_words]) for 
                   context_word,target_words in cbows]

print(cbow_vector_pairs[0][0]) #contains 6 context words
print(cbow_vector_pairs[0][1]) #contains tragte words
# %%
#Sum the context vector to get single context vector
cbow_vector_pairs=[(torch.sum(torch.stack(context_words),dim=0),target_words) for context_words,target_words in cbow_vector_pairs]
print(cbow_vector_pairs[0])
# %%
#Converting Dataset to pytorch Dataset
class CustomDataset(Dataset):
    def __init__(self,data):
        self.inputs=[item[0] for item in data]
        self.outputs=[item[1] for item in data]

    def __len__(self):
        return len(self.inputs)
    def __getitem__(self, index):
        input_sample=self.inputs[index]
        output_sample=self.outputs[index]
        return input_sample,output_sample
# %%
#Training
cbow_vector_pairs =random.sample(cbow_vector_pairs,len(cbow_vector_pairs))
split_index=int(len(cbow_vector_pairs)*0.90)

train_data=CustomDataset(cbow_vector_pairs[:split_index])
test_data=CustomDataset(cbow_vector_pairs[split_index:])
batch_size=64
train_dataloader=DataLoader(train_data,batch_size=batch_size,shuffle=True)
validation_dataloader =DataLoader(test_data,batch_size=batch_size,shuffle=True)
# %%
class word2vec(nn.Module):
    def __init__(self, vocab_size,vector_dim):
        super().__init__()
        self.vocab_size=vocab_size
        self.vector_dim=vector_dim
        self.w1=nn.Parameter(data=torch.randn(self.vocab_size,self.vector_dim),requires_grad=True)
        self.w2=nn.Parameter(data=torch.randn(self.vector_dim,self.vocab_size),requires_grad=True)

    def forward(self,X):
        X=X@self.w1
        X=X@self.w2
        return X
vocab_size=len(unique_words)
vector_dim=2
model=word2vec(vocab_size,vector_dim)

def train(model,train_dataloader,validation_dataloader,epochs,learning_rate,verbose=True):
    loss_fn=nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(params=model.parameters(),lr=learning_rate)
    trains_set_log=[]
    validation_set_log=[]
    for epoch in range(epochs):
        if verbose: print("Epoch: ", epoch)
        model.train()
        total_train_loss = 0.0
        num_train_batches = 0
        for input_batch,output_batch in train_dataloader:
            y_logit=model(input_batch)
            train_loss=loss_fn(y_logit,output_batch)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            total_train_loss +=train_loss.item()
            num_train_batches+=1
        average_train_loss=total_train_loss / num_train_batches
        trains_set_log.append(average_train_loss)

        model.eval()
        total_validation_loss = 0.0
        num_validation_batches = 0

        with torch.inference_mode():
            for input_batch,output_batch in validation_dataloader:
                y_val_logits=model(input_batch)
                validation_loss=loss_fn(y_val_logits,output_batch)
                total_validation_loss += validation_loss.item()
                num_validation_batches += 1
        average_validation_loss = total_validation_loss / num_validation_batches
        validation_set_log.append(average_validation_loss)
        if verbose: print("Train Loss: ", average_train_loss, "|||", "Validation Loss: ", average_validation_loss)
    
    return model, trains_set_log, validation_set_log
model, train_set_loss_log, validation_set_loss_log = train(model, train_dataloader, validation_dataloader, 
                                                                 epochs=3, learning_rate=0.01, verbose=True)

plt.plot(train_set_loss_log, color='red', label='train_loss')
plt.plot(validation_set_loss_log, color='blue', label='validation_loss')

plt.title("Loss During Training")
plt.xlabel("Epoch")
plt.ylabel("Cross Entropy")
plt.legend()
plt.savefig("train_test_loss.jpg")
# %%
# Word Vectors
params = list(model.parameters())
word_vectors = params[0].detach()
# Create a dictionary with the same order mapping
word_dict = {word: vector for word, vector in zip(unique_words, word_vectors)}

def cosine_similarity(v1,v2):
    return (v1@v2)/(torch.norm(v1)*torch.norm(v2))

def most_similar(word,word_dict,top_k=5):
    if word not in word_dict:
        raise ValueError(f"{word} not found in word dictionary")
    querry_vector=word_dict[word]
    similarity={}
    for other_word, other_vector in word_dict.items():
        if other_word != word:
            similar= cosine_similarity(querry_vector, other_vector)
            similarity[other_word] = similar

    # Sort the words by similarity in descending order
    sorted_similarities = sorted(similarity.items(), key=lambda x: x[1], reverse=True)

    # Get the top-k most similar words
    top_similar_words = sorted_similarities[:top_k]

    return top_similar_words
print(word_dict)
x_coords, y_coords = zip(*[word_dict[word].numpy() for word in list(word_dict.keys())])

plt.figure(figsize=(10, 8))
plt.scatter(x_coords, y_coords, marker='o', color='blue')

for i, word in enumerate(list(word_dict.keys())):
    plt.annotate(word, (x_coords[i], y_coords[i]), textcoords="offset points", xytext=(0, 5), ha='center')

plt.title('Word Embeddings')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.grid(True)
plt.savefig("word_embedding.jpg")