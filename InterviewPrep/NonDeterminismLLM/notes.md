Asking LLMs like ChatGPT same question multiple times provides different results. Why?
Even after adjusting the temperature to 0 which by theory should produce deterministic or same results.
LLMs predict next word from matrix multiplication by attention mechanism which results from converting words to embeddings on a high level.
What is determinism?
A deterministic algorithm is an algorithm that, given a particular input, will always produce the same output


One common hypothesis is floating-point non-associativity and concurrent execution leads to nondeterminism.

Floating-point arithmetic in GPUs exhibits non-associativity, meaning 

(a+b)+c != a+(b+c) due to finite precision and rounding errors

(0.1 + 1e20) - 1e20
>>> 0
0.1 + (1e20 - 1e20)
>>> 0.1

Floating-point numbers are useful because they allow dynamic precision.
Example
3460 can be represented as 3.46*1e3
0.456 as 4.56*1e-1
This way both large and small numbers can be represented. But when we add 2 floating numbers with different exponents, there is information loss.
import random

vals = [1e-10, 1e-5, 1e-2, 1]
vals = vals + [-v for v in vals]

results = []
random.seed(42)
for _ in range(10000):
    random.shuffle(vals)
    results.append(sum(vals))

results = sorted(set(results))
print(f"There are {len(results)} unique results: {results}")

>>Output:
There are 102 unique results: [-8.326672684688674e-17, -7.45931094670027e-17, ..., 8.326672684688674e-17]

But still this does answer our question. That’s where we see how kernels are implemented.

The order in which concurrent threads finish is non deterministic and the accumulation order depends on the order in which concurrent threads finish, which leads to nondeterminism.

 A GPU launches a program concurrently across many “cores” (i.e. SMs).
As the cores have no inherent synchronization among them, this poses a challenge if the cores need to communicate among each other. 
For example, if all cores must accumulate to the same element, you can use an “atomic add” . The atomic add is “nondeterministic” — the order in which the results accumulate is purely dependent on which core finishes first.

Suppose you have 100 element vector and you need to perform sum operation. You have 100 cores, you can load elements in parallel but we need to reduce down to single element. This is where atomic add comes in, but the order in which these elements are added is not guaranteed, since the order of execution of threads is different.  
 
Hence when you execute the same kernel twice you get non determinism

Atomic add definition->Fetch-and-add performs the following operation: increment the value at address x by a, where x is a memory location and a is some value, and return the original value at x.


Surprisingly in forward pass of LLM there is no atomic add present 
To overcome this neural network libraries have adopted various strategies.

But this itself doesn't make the whole system deterministic. Since there are multiple users, there requests can be grouped together on the server and we don't know if kernel operations are batch invariant.

Example
import torch
torch.set_default_device('cuda') 

B = 2048
D = 4096
a = torch.linspace(-1000, 1000, B*D).reshape(B, D)
b = torch.linspace(-1000, 1000, D*D).reshape(D, D)
Doing a matrix vector multiplication by taking
the first element of the batch

out1 = torch.mm(a[:1], b)
Doing a matrix matrix multiplication and then taking
the first element of the batch

out2 = torch.mm(a, b)[:1]
print((out1 - out2).abs().max()) # tensor(1669.2500, device='cuda:0')

When you make a query to an inference endpoint, the amount of load the server is under is effectively “nondeterministic” from the user’s perspective. The load determines the batch size that the kernels are run under, and thus changes the eventual result of each individual request!

Hence we need to implement following operations- RMSNorm, matrix multiplication and attention that involves reduction.

To overcome them we need data parallelism strategies in each of the above operations.
