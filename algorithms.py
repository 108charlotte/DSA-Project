import numpy as np
import pandas as pd
import neurokit2 as nk

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
    data = pd.read_csv('hr_data.csv')

    time_series = data.iloc[:, 0].to_numpy()
    ppg_values = data.iloc[:, 1].to_numpy()

    fs = int(np.round(np.median(time_series[1:] - time_series[:-1])))

    start_ms = data.iloc[0, 0]
    end_ms   = data.iloc[-1, 0]
    start_s = int(start_ms / 1000)
    end_s   = int(end_ms / 1000)
    proc_start = 0
    proc_end = end_s - start_s

    signals, info = nk.ppg_process(ppg_values[proc_start*fs:proc_end*fs], sampling_rate=fs)

    signals, info = nk.ppg_process(ppg_values, sampling_rate=fs)
    HR = signals['PPG_Rate']
    HR.to_csv('HeartRate.csv')

    data = pd.read_csv('/content/HeartRate.csv')
    string = input
    total_secs = []
    minutes = 0
    seconds = 0
    string_length = len(string)
    midpoint = string_length // 2
    minutes_str = string[:midpoint]
    seconds_str = string[midpoint:]
    for char in minutes_str:
        minutes += ord(char)
    for char in seconds_str:
        seconds += ord(char)
    total_secs.append(minutes*60 + seconds)
    key1 = data.iloc[string, 1]
    key2 = int((key1*10000))
    return key2

def hash_3(input): 
    return 0

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

words = 100000

hash_table.populate(1, file_content, words)
print(hash_table)
print("Duplicates: " + str(hash_table.duplicates))
print("Collisions: " + str(hash_table.num_collisions))
print("Entries: " + str(hash_table.num_elements))