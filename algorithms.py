import numpy as np
import pandas as pd
import neurokit2 as nk
import sys
import os
import time
import requests
from urllib.parse import quote_plus

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
    # Use `wordfreq` zipf frequencies as a free, local proxy for word popularity.
    # This avoids external APIs and quotas. Install with: `pip install wordfreq`.
    query = str(input).strip()
    if not query:
        return 0

    # simple per-process cache
    if not hasattr(hash_4, '_cache'):
        hash_4._cache = {}

    if query in hash_4._cache:
        return hash_4._cache[query]

    try:
        from wordfreq import zipf_frequency
    except Exception:
        raise RuntimeError("hash_4 requires the 'wordfreq' package")

    # lowercase for normalization; wordfreq expects language code 'en'
    z = zipf_frequency(query.lower(), 'en')
    # zipf_frequency returns a float that may be negative for very rare words.
    score = max(0, int(z * 100))
    hash_4._cache[query] = score
    return score

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
        
        self.max_hashed_in_slot = -float('inf')
        self.min_hashed_in_slot = float('inf')
        self.duplicates = 0

        self.num_elements = 0

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
            # update min/max in-slot counts
            cnt = 1
            if self.max_in_slot == -float('inf') or cnt > self.max_in_slot:
                self.max_in_slot = cnt
            if self.min_in_slot == float('inf') or cnt < self.min_in_slot:
                self.min_in_slot = cnt
            return
        cur = current
        while cur:
            # if duplicate then count as duplicate and don't add to table
            if cur.value == value:
                self.duplicates += 1
                return
            # continues thru node list
            cur = cur.next

        # collision
        new_node = Node(hashed, value)
        new_node.next = self.table[index]
        self.table[index] = new_node
        self.size += 1
        self.num_collisions += 1
        self.num_elements += 1
        
        # generated by copilot to update min & max
        cnt = 0
        cur = self.table[index]
        while cur:
            cnt += 1
            cur = cur.next
        if self.max_in_slot == -float('inf') or cnt > self.max_in_slot:
            self.max_in_slot = cnt
        if self.min_in_slot == float('inf') or cnt < self.min_in_slot:
            self.min_in_slot = cnt
        
        cur = self.table[index]
        slot_max_hash = -float('inf')
        slot_min_hash = float('inf')
        while cur:
            if cur.key > slot_max_hash:
                slot_max_hash = cur.key
            if cur.key < slot_min_hash:
                slot_min_hash = cur.key
            cur = cur.next
        if self.max_hashed_in_slot == -float('inf') or slot_max_hash > self.max_hashed_in_slot:
            self.max_hashed_in_slot = slot_max_hash
        if self.min_hashed_in_slot == float('inf') or slot_min_hash < self.min_hashed_in_slot:
            self.min_hashed_in_slot = slot_min_hash

    # inspired by article again
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
        # string pattern recognition generated by copilot to determine whether or not smth is a character in a word
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

# set hash table size here (reminder to myself)
hash_table = HashTable(1024)

# load string from file
file_path = 'generated_words.txt'

with open(file_path, 'r') as file:
    file_content = file.read()

# used copilot to make this take conditional arguments
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

# used copilot to fix max and min count displays
# Recompute per-slot counts (including empty slots) and update min/max counts.
counts = []
for i in range(hash_table.capacity):
    cnt = 0
    cur = hash_table.table[i]
    while cur:
        cnt += 1
        cur = cur.next
    counts.append(cnt)

hash_table.max_in_slot = max(counts) if counts else 0
hash_table.min_in_slot = min(counts) if counts else 0
# me adding on mean calculation
# print("Mean: " + str(sum(counts) / len(counts) if counts else 0))

print("Max in one slot:", hash_table.max_in_slot)
print("Min in one slot:", hash_table.min_in_slot)