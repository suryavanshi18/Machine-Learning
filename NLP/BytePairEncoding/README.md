# Byte Pair Encoding (BPE)

Byte Pair Encoding (BPE) is a simple yet powerful text compression and tokenization algorithm. It’s widely used in modern natural language processing pipelines, including models like GPT-2 and GPT-3.

---

## Setting Up Your Python Environment

To keep dependencies isolated, use a Python virtual environment:

Create a virtual environment named sampleEnv
python -m venv sampleEnv

Activate the virtual environment:
On Windows:
.\sampleEnv\Scripts\Activate.ps1

On macOS/Linux:
source sampleEnv/bin/activate

---

## BPE Algorithm Outline

### 1. Identify Frequent Pairs
- In each iteration, scan the text to find the most commonly occurring pair of bytes (or characters).

### 2. Replace and Record
- Replace that pair everywhere it occurs with a new placeholder ID (one not already in use).
  - Example: If your byte IDs start from 0–255, the first new token will be 256.
- Record this mapping (pair → new ID) in a lookup table.
- The maximum size of the lookup table (the “vocabulary size”) is a hyperparameter.
  - For reference: GPT-2 uses a vocabulary size of 50,257.

### 3. Repeat Until No Gains
- Keep repeating steps 1 and 2, merging the most frequent pairs.
- Stop when no pair occurs more than once or your lookup table reaches the desired vocabulary size.

---

## Decompression (Decoding)

To restore the original text:
- Reverse the process by substituting each placeholder ID with its corresponding original pair using the lookup table.

---