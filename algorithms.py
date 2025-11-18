import numpy as np
import pandas as pd
import neurokit2 as nk
import sys

# Cache the processed heart-rate series so we don't recompute it on every hash call
_HR_CACHE = None

def _get_hr_series():
    global _HR_CACHE
    if _HR_CACHE is not None:
        return _HR_CACHE

    # Load raw PPG/HR data and compute HR series once
    raw = pd.read_csv('hr_data.csv')
    time_series = raw.iloc[:, 0].to_numpy()
    ppg_values = raw.iloc[:, 1].to_numpy()
    # estimate sampling frequency from timestamps (assumes ms timestamps)
    diffs = time_series[1:] - time_series[:-1]
    if len(diffs) == 0:
        raise ValueError('hr_data.csv has insufficient rows')
    fs = int(np.round(np.median(diffs)))
    # neurokit expects sampling_rate in Hz; if timestamps are ms convert to Hz
    # if fs looks like milliseconds-per-sample, convert to samples-per-second
    if fs > 1000:  # heuristic: timestamps likely in ms
        sampling_rate = int(round(1000.0 / fs))
    else:
        sampling_rate = int(fs) if fs > 0 else 1

    signals, info = nk.ppg_process(ppg_values, sampling_rate=sampling_rate)
    HR = signals['PPG_Rate'].reset_index(drop=True)
    _HR_CACHE = HR
    return _HR_CACHE

def to_index(hashed_val, len): 
    return hashed_val % len

def hash_1(input): 
    sum = 0
    letters = []
    for letter in input: 
        num = ord(letter)
        # letter already appeared
        if letter in letters: 
            num += 1
        letters.append(letter)
        sum += num
    # Return the computed sum directly. Multiplying by 1000 made every
    # hash a multiple of 1000, and since the table capacity (50)
    # divides 1000, every value mapped to the same index (hash % 50 == 0).
    return sum

def hash_2(input): 
    sum = 0
    for letter in input: 
        num = ord(letter)
        sum += num
    sum *= 10**30
    sum %= 1_999_999_999
    sum -= 999_999_937
    return sum


def hash_3(input): 
    # Use the cached HR series (computed once). Convert the input string
    # to an integer index safely and use modulo to stay within bounds.
    HR = _get_hr_series()

    s = str(input)
    if len(s) == 0:
        idx = 0
    else:
        midpoint = len(s) // 2
        minutes = sum(ord(c) for c in s[:midpoint])
        seconds = sum(ord(c) for c in s[midpoint:])
        total_secs = minutes * 60 + seconds
        # Map the potentially large number into the HR series range
        idx = int(total_secs % len(HR))

    key1 = HR.iloc[idx]
    # Ensure numeric and convert to integer hashable value
    try:
        key2 = int(float(key1) * 10000)
    except Exception:
        # fallback: use idx if HR value cannot be converted
        key2 = int(idx)
    return key2

def hash_4(input): 
    return 0

# inspired by https://www.geeksforgeeks.org/dsa/implementation-of-hash-table-in-python-using-separate-chaining/

class Node:
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.next = None

class HashTable:
    def __init__(self, capacity):
        self.capacity = capacity
        self.size = 0
        self.table = [None] * capacity
        
        self.num_collisions = 0
        self.max_in_slot = -float('inf')
        self.min_in_slot = float('inf')
        self.duplicates = 0

        self.num_elements = 0
        '''
        self.range = 0
        self.standard_deviation = 
        '''

    def _hash(self, num, key): 
        match (num): 
            case 1: 
                return hash_1(key)
            case 2: 
                return hash_2(key)
            case 3: 
                return hash_3(key)
            case 4: 
                return hash_4(key)
    
    def _insert(self, num, key, value): 
        hashed = self._hash(num, key)
        index = to_index(hashed, self.capacity)
        current = self.table[index]
        # empty slot: insert directly
        if current is None:
            self.table[index] = Node(hashed, value)
            self.size += 1
            self.num_elements += 1
            return

        # non-empty slot: check for duplicate first
        cur = current
        while cur:
            # Treat as a duplicate only when the same word (value) already exists in the chain.
            # Previously this compared hashes (`cur.key == hashed`) which treats different
            # words that collide to the same hash as duplicates. That produced false positives.
            if cur.value == value:
                self.duplicates += 1
                return
            cur = cur.next

        # no duplicate found -> insert new node (this is a collision)
        new_node = Node(hashed, value)
        new_node.next = self.table[index]
        self.table[index] = new_node
        self.size += 1
        self.num_collisions += 1
        self.num_elements += 1
    
    # (add later, don't want this to interfere w/ run times) def resize(self, new_capacity): 
        # self.capacity = new_capacity

    def search(self, num, key):
        # compute hashed index then search by stored word value
        hashed = self._hash(num, key)
        index = to_index(hashed, self.capacity)

        current = self.table[index]
        while current:
            if current.value == key:
                return current.value
            current = current.next
        raise KeyError(key)

    def populate(self, num, text, num_elements): 
        # Split the input text into alphabetic words and insert up to `num_elements` words.
        # This fixes the previous bug where `num_elements` was treated as a character count.
        import re
        words = re.findall(r"[A-Za-z]+", text)
        # If num_elements is None or larger than available words, insert all words
        limit = num_elements if num_elements is not None else len(words)
        limit = min(limit, len(words))
        for i in range(limit):
            w = words[i]
            if w:  # avoid inserting empty strings
                self._insert(num, w, w)
    
    def __str__(self):
        # Return a compact table of index -> number of entries in that slot.
        lines = []
        lines.append("Index | Count")
        lines.append("----- | -----")
        for i in range(self.capacity):
            cnt = 0
            current = self.table[i]
            while current:
                cnt += 1
                current = current.next
            lines.append(f"{i:5} | {cnt}")
        return "\n".join(lines)


hash_table = HashTable(4)

# load string from file
file_path = 'generated_words.txt'

with open(file_path, 'r') as file:
    file_content = file.read()

# Defaults
hash_num = 1       # which hash function to use (1..4)
words = 100000     # number of words to insert

# Command-line positional handling:
# - `python3 algorithms.py 2` -> use hash function 2
# - `python3 algorithms.py 2 210` -> use hash 2 and insert 210 words
# - `python3 algorithms.py 210` -> if first arg not in 1..4, treat as words count
if len(sys.argv) >= 2:
    try:
        first = int(sys.argv[1])
        if 1 <= first <= 4:
            hash_num = first
            if len(sys.argv) >= 3:
                try:
                    words = int(sys.argv[2])
                except ValueError:
                    print(f"Warning: second positional argument '{sys.argv[2]}' is not an integer; using default words={words}")
        else:
            # treat first positional argument as words count
            words = first
    except ValueError:
        print(f"Warning: positional argument '{sys.argv[1]}' is not an integer; using defaults hash={hash_num}, words={words}")

hash_table.populate(hash_num, file_content, words)
print(hash_table)
print("Duplicates: " + str(hash_table.duplicates))
print("Collisions: " + str(hash_table.num_collisions))
print("Entries: " + str(hash_table.num_elements))