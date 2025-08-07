from collections import Counter,deque
from functools import lru_cache
import json

#To activate virtual env command-> .\sampleEnv\Scripts\Activate.ps1

#Bytearray creates mutable sequence of bytes which is an array from 0 to 255
# text = "This is some text"
# byte_ary = bytearray(text, "utf-8")
# print(byte_ary)
# byte_ary[1]=110
# print(byte_ary)

#creates an array of that size, initialized with null bytes \x00
# ba=bytearray(2)
# print(ba)
# ba[0]=65
# print(ba)

#ba = bytearray([72, 101, 108, 108, 111])
#The list [72, 101, 108, 108, 111] corresponds to the ASCII values for "Hello"
#print(ba)

# text="Hello World! Welcome to Pune"
# byte_arry=bytearray(text,"utf-8")
# #creates id for each character
# ids=list(byte_arry)
# print(ids)

# t=[]
# text="Hello World"
# for i,char in enumerate(text):
#     if char==' ' and i!=0:
#             t.append("Ġ")
#     if char!=' ':
#         t.append(char)
# t="".join(t)
# c=[chr(i) for i in range(256)]
# print(c)
# allowed_special={"<|endoftext|>"}
# for token in allowed_special:
#     print(token)

# a=[1,2,3,6,1,2,4]
# print(a)
# print(a[1:])
# print(dict(zip(a,a[1:])))

'''
O/P
[1, 2, 3, 6, 1, 2, 4]
[2, 3, 6, 1, 2, 4]
{1: 2, 2: 4, 3: 6, 6: 1}

Notes:

The first 2 key is created with value 3 from pair (2, 3).
But when the second (2, 4) pair appears later, it overwrites the previous 
key 2's value 3 with 4.

'''
# a=[1,2,3,6,1,2,3]
# print(Counter(zip(a,a[1:])))





class BPETokenizer:
    def __init__(self):
        self.vocab={}
        self.inverse_vocab={}
        self.bep_merges={}
    
    def train(self,text,vocab_size,allowed_special={"<endoftext>"}):
        processed_text=[]
        for i,char in enumerate(text):
            if char==' ' and i!=0:
                processed_text.append("Ġ")
            if char!=' ':
                processed_text.append(char)
        processed_text="".join(processed_text)

        unique_chars=[chr(i) for i in range(256)]
        unique_chars.extend(char for char in sorted(set(processed_text)) if char not in unique_chars)

        if "Ġ" not in unique_chars:
            unique_chars.append("Ġ")

        self.vocab={i:char for i,char in enumerate(unique_chars)}
        self.inverse_vocab={char:i for i,char in enumerate(unique_chars)}

        if allowed_special:
            for token in allowed_special:
                if token not in self.inverse_vocab:
                    new_id=len(self.vocab)
                    self.vocab[new_id]=token
                    self.inverse_vocab[token]=new_id
        
        token_ids=[self.inverse_vocab[char] for char in processed_text]

        for new_id in range(len(self.vocab),vocab_size):
            pair_id=self.find_freq_pair(token_ids)
            if pair_id is None:
                break
            token_ids=self.replace_pair(token_ids,pair_id,new_id)
            self.bep_merges[pair_id]=new_id

        for (p0,p1),new_id in self.bep_merges.items():
            merged_token=self.vocab[p0]+self.vocab[p1]
            self.vocab[new_id]=merged_token
            self.inverse_vocab[merged_token]=new_id
    
    @staticmethod
    def find_freq_pair(token_ids):
        """
        Zip maps corresponding elements of the 2 dictionary
       
        Example:
        a=[1,2,3,6,1,2,4]
        print(a)
        print(a[1:])
        print(dict(zip(a,a[1:])))

        O/P
        [1, 2, 3, 6, 1, 2, 4]
        [2, 3, 6, 1, 2, 4]
        {1: 2, 2: 4, 3: 6, 6: 1}

        1.The first 2 key is created with value 3 from pair (2, 3).
          But when the second (2, 4) pair appears later, it overwrites the previous 
          key 2's value 3 with 4.

        2.If we have zip(a,a[2:]) 
          Then we would map current element with second element from curr


        pairs ->Counter({(1, 2): 2, (2, 3): 2, (3, 6): 1, (6, 1): 1})
        we take the max of value in dict and return the corresponding key
        """
        pairs=Counter(zip(token_ids,token_ids[1:]))
        return max(pairs.items(), key=lambda x:x[1])[0]
    
    @staticmethod
    def replace_pair(token_ids,pair_id,new_id):
        '''
        Deque is double ended queue
        We pop elements from left side
        curr is leftmost element
        dp[0] is second leftmost element
        freq_pair method returns us the tokenids which occur freq
        Hence if curr and dq[0] are those pairs then we pop dq[0] as well
        and replace it with new_id
        '''
        dq=deque(token_ids)
        new_token_ids=[]

        while dq:
            curr=dq.popleft()
            if dq and (curr,dq[0])==pair_id:
                new_token_ids.append(new_id)
                dq.popleft()
            else:
                new_token_ids.append(curr)
        return new_token_ids
    
    def load_vocab_and_merges_from_openai(self,vocab_path,bpe_merges_path):

        with open(vocab_path,"r",encoding="utf-8") as file:
            loaded_vocab=  json.load(file)
            self.vocab={int(v):k for k,v in loaded_vocab.items()}
            self.inverse_vocab={k:int(v) for k,v in loaded_vocab.items()}

        with open(bpe_merges_path,"r",encoding="utf-8") as file:
            lines=file.readlines()
            #skip first line or the header 
            if lines and lines[0].startswith("#"):
                lines=lines[1:]
            
            for rank,line in enumerate(lines):
                pair=tuple(line.strip().split())

                if(len!=2):
                    print(f" line {rank+1} has more than 2 entries")
                    continue
                token1,token2=pair
                if token1 in self.inverse_vocab and token2 in self.inverse_vocab:
                    token1_id=self.inverse_vocab[token1]
                    token2_id=self.inverse_vocab[token2]
                    merged_token=token1_id+token2_id
                    if merged_token in self.inverse_vocab:
                        merged_token_id=self.inverse_vocab[merged_token]
                        self.bep_merges[(token1,token2)]=merged_token_id
                    else:
                        print(f"Merged token {merged_token} not found")
                else:
                    print(f"Skipping pair {pair} not found in vocab")

    def save_vocab_and_merges(self,vocab_path,bpe_merges_path):
        with open(vocab_path,"w",encoding="utf-8") as file:
            json.dump({k:v for k,v in self.vocab.items()},file,ensure_ascii=False,indent=2)

        with open(bpe_merges_path,"w",encoding="utf-8") as file:
            merges_list=[{"pair":list(pair),"new_id":new_id} for pair,new_id in self.bep_merges.items()]   
            json.dumps(merges_list,ensure_ascii=False,indent=2)

    def load_vocab_and_merges(self,vocab_path,bpe_merges_path):
        with open(vocab_path,"r",encoding="utf-8") as file:
            loaded_vocab=json.load(file)
            self.vocab={int(k):v for k,v in loaded_vocab.items()}
            self.inverse_vocab={v:int(k) for k,v in loaded_vocab.items()}
        
        with open(bpe_merges_path,"r",encoding="utf-8") as file:
            merges_list=json.load(file)
            for merge in merges_list:
                pair=tuple(merge["pair"])
                new_id=merge["new_id"]
                self.bep_merges[pair]=new_id
    @lru_cache
    def get_special_token_id(self,token):
        return self.inverse_vocab.get(token,None)    

    def encode(self,text):
        tokens=[]
        words=text.replace("\n"," \n ").split()
        for i,word in enumerate(words):
            if i>0 and not word.startswith("\n"):
                tokens.append("Ġ"+word)
            else:
                tokens.append(word)
        token_ids=[]
        for token in tokens:
            if token in self.inverse_vocab:
                token_id=self.inverse_vocab[token]
                token_ids.append(token_id)
            else:
                sub_token_ids=self.tokenize_with_bpe(token)
                token_ids.extend(sub_token_ids)
        return token_ids
    def tokenize_with_bpe(self,token):
        token_ids=[self.inverse_vocab.get(char,None) for char in token]
        if None in token_ids:
            raise ValueError(f"Charachters not found in vocab!")
        can_merge=True
        new_tokens=[]
        while can_merge and len(token_ids)>1:
            can_merge=False
           
            i=0
            while i<len(token_ids)-1:
                pair=(token_ids[i],token_ids[i+1])
                if pair in self.bep_merges:
                    merged_token_id=self.bep_merges[pair]
                    new_tokens.append(merged_token_id)
                    i+=2
                    can_merge=True
                else:
                    new_tokens.append(token_ids[i])
                    i+=1
        if i<len(token_ids):
            new_tokens.append(token_ids[i])
        token_ids=new_tokens

    def decode(self,token_ids):
        ans=""
        for token_id in token_ids:
            if token_id not in self.vocab:
                raise ValueError(f"Token ID not found")
            token=self.vocab[token_id]
            if token.startswith("Ġ"):
                ans+=" "+token[1:]
            else:
                ans+=token
        return ans
              
import os
import urllib.request

if not os.path.exists("the-verdict.txt"):
    url = ("https://raw.githubusercontent.com/rasbt/"
           "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
           "the-verdict.txt")
    file_path = "the-verdict.txt"
    urllib.request.urlretrieve(url, file_path)

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    text = f.read()
tokenizer = BPETokenizer()
tokenizer.train(text, vocab_size=1000, allowed_special={"<|endoftext|>"})

print(len(tokenizer.vocab))
print(len(tokenizer.bep_merges))

input_text = "Jack embraced beauty through art and life."
token_ids = tokenizer.encode(input_text)
print(token_ids)
print(tokenizer.decode(token_ids))
tokenizer.save_vocab_and_merges(vocab_path="vocab.json", bpe_merges_path="bpe_merges.txt")
tokenizer2 = BPETokenizer()
tokenizer2.load_vocab_and_merges(vocab_path="vocab.json", bpe_merges_path="bpe_merges.txt") 
print(tokenizer2.decode(token_ids))   
