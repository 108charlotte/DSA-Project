import numpy as np
import pandas as pd
import neurokit2 as nk
import sys
import argparse
from urllib.parse import quote_plus
import statistics
import csv
import math

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
        sum = int(sum*math.pi) 
        sum *= 997
    # credit to chatgpt for the following line
    #sets min to zero, mods the number to be within integer range, then shifts over to -2B to 2B 
    return (sum - (-2_147_483_647)) % 4_294_967_296 + -2_147_483_648
    
def hash_2(input): 
    sum = 0
    for letter in input: 
        num = ord(letter)
        sum += num
    sum *= 10**30 - 6
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

# used copilot to generate, came up with idea ourselves
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

    # lowercase for normalization; wordfreq expects language code 'en'
    z = zipf_frequency(query.lower(), 'en')
    # zipf_frequency returns a float that may be negative for very rare words.
    score = max(0, int(z * 100))
    hash_4._cache[query] = score
    # returns numerical value corresponding to the frequency of a word in the english language
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
        # per-slot counts to help compute min/max chain lengths quickly
        self.counts = [0] * capacity

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

        # update hashed min/max tracking
        try:
            if hashed > self.max_hashed_in_slot:
                self.max_hashed_in_slot = hashed
            if hashed < self.min_hashed_in_slot:
                self.min_hashed_in_slot = hashed
        except Exception:
            pass

        # empty slot: insert directly
        if current is None:
            self.table[index] = Node(hashed, value)
            self.size += 1
            self.num_elements += 1
            self.counts[index] = 1
        else:
            # collision: append to chain, but avoid duplicates
            self.num_collisions += 1
            node = current
            while node is not None:
                if node.value == value:
                    # duplicate entry (same original key) — skip
                    self.duplicates += 1
                    return
                if node.next is None:
                    break
                node = node.next
            # append new node
            node.next = Node(hashed, value)
            self.size += 1
            self.num_elements += 1
            self.counts[index] += 1

        # update slot-level min/max for non-empty slots
        nonzero = [c for c in self.counts if c > 0]
        if nonzero:
            self.max_in_slot = max(nonzero)
            self.min_in_slot = min(nonzero)
        else:
            self.max_in_slot = 0
            self.min_in_slot = 0

    def insert(self, num, key):
        # public insert helper: value stored is the original key (for duplicate checks)
        return self._insert(num, key, key)

    def stats(self):
        return {
            'capacity': self.capacity,
            'num_elements': self.num_elements,
            'num_collisions': self.num_collisions,
            'duplicates': self.duplicates,
            'max_in_slot': self.max_in_slot,
            'min_in_slot': self.min_in_slot,
            'max_hashed_in_slot': self.max_hashed_in_slot,
            'min_hashed_in_slot': self.min_hashed_in_slot,
        }
    
    def __str__(self):
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


file_path = 'generated_words.txt'

with open(file_path, 'r') as file:
    file_content = file.read()

# Pre-split full word list once for chunking
import re
all_words = re.findall(r"[A-Za-z]+", file_content)


# used copilot to make mass hash table insertion able to run from terminal and export data to csv, can either run with generate_csv to get stats for table values in an array or compare to get the count per slot of every function

# Defaults
hash_num = 1       # which hash function to use (1..4)
words = 100000     # number of words to insert

TABLE_SIZES = [4, 7, 8, 16, 17, 31, 32, 64, 67, 128, 256, 257, 509, 512, 1021, 1024]
# can specify which chunk of entries to use, each len = 5000 words

# written by copilot to automate process of recording entries, max in one slot, and min in one slot into csv with changing table size. 
# i prompted it to use my previous code and previously calculated values

# Use argparse to require explicit options. Nothing runs unless the user supplies
# all required parameters explicitly. Two subcommands: `run` and `generate_csv`.

def _get_words_for_chunk(chunk, chunk_size=5000):
    start = (chunk - 1) * chunk_size
    end = start + chunk_size
    return all_words[start:end]


def generate_csv(hash_num, chunk, out_path='hash_results.csv'):
    words_slice = _get_words_for_chunk(chunk)
    if len(words_slice) == 0:
        raise ValueError('Selected chunk has no words')

    rows = []
    for size in TABLE_SIZES:
        ht = HashTable(size)
        for w in words_slice:
            try:
                ht.insert(hash_num, w)
            except RuntimeError as e:
                raise
        stats = ht.stats()
        stats_row = {
            'table_size': stats['capacity'],
            'elements_inserted': stats['num_elements'],
            'collisions': stats['num_collisions'],
            'duplicates': stats['duplicates'],
            'max_in_slot': stats['max_in_slot'],
            'min_in_slot': stats['min_in_slot'],
            'max_hashed_in_slot': stats['max_hashed_in_slot'],
            'min_hashed_in_slot': stats['min_hashed_in_slot'],
            'stddev': 0,
        }
        # compute standard deviation across all slots (including zeros)
        try:
            sd = statistics.stdev(ht.counts) if len(ht.counts) > 1 else 0
        except Exception:
            sd = 0
        stats_row['stddev'] = sd
        rows.append(stats_row)

    # write CSV
    fieldnames = [
        'table_size', 'elements_inserted', 'collisions', 'duplicates',
        'max_in_slot', 'min_in_slot', 'max_hashed_in_slot', 'min_hashed_in_slot'
    ]
    # include stddev column
    fieldnames.append('stddev')
    with open(out_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    return out_path

def compare_hashes(table_size, chunk, out_path='compare_hashes.csv'):
    """Run the 'run' experiment for each hash function (1..4) at a single
    `table_size` and write a CSV where each row is a hash number and each
    column is an index in the table containing the count for that index.
    """
    words_slice = _get_words_for_chunk(chunk)
    if len(words_slice) == 0:
        raise ValueError('Selected chunk has no words')

    # collect counts per hash function
    results = {}
    for h in range(1, 5):
        ht = HashTable(table_size)
        for w in words_slice:
            ht.insert(h, w)
        # ensure a plain list copy
        results[h] = list(ht.counts)

    # write CSV: header = ['hash function/slot #', '0', '1', ...]
    header = ['hash'] + [str(i) for i in range(table_size)]
    with open(out_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        for h in range(1, 5):
            row = [h] + results[h]
            writer.writerow(row)

    return out_path

def main():
    # Quick mode: if called as `python algorithms.py <hash> <chunk> [table_size]`
    # run both the single run (printing table/stats and writing table CSV)
    # and generate the CSV over TABLE_SIZES. This keeps the interface terse.
    if len(sys.argv) >= 3:
        try:
            maybe_hash = int(sys.argv[1])
            maybe_chunk = int(sys.argv[2])
            if 1 <= maybe_hash <= 4 and maybe_chunk in (1, 2):
                # optional table size as third arg
                table_size = None
                if len(sys.argv) >= 4:
                    try:
                        table_size = int(sys.argv[3])
                    except Exception:
                        table_size = None

                # run single table-size experiment
                words_slice = _get_words_for_chunk(maybe_chunk)
                tsz = table_size if table_size is not None else TABLE_SIZES[0]
                ht = HashTable(tsz)
                for w in words_slice:
                    ht.insert(maybe_hash, w)
                stats = ht.stats()
                try:
                    sd = statistics.stdev(ht.counts) if len(ht.counts) > 1 else 0
                except Exception:
                    sd = 0
                stats['stddev'] = sd
                print(ht)
                print('Run stats:', stats)
                # write the table CSV
                out_table_csv = f'table_hash{maybe_hash}_chunk{maybe_chunk}_size{ht.capacity}.csv'
                with open(out_table_csv, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['Index', 'Count'])
                    for i, c in enumerate(ht.counts):
                        writer.writerow([i, c])
                print(f'Wrote table CSV to: {out_table_csv}')

                # generate CSV over TABLE_SIZES
                out = generate_csv(maybe_hash, maybe_chunk, out_path=f'hash_results_hash{maybe_hash}_chunk{maybe_chunk}.csv')
                print(f'Wrote CSV to: {out}')
                return
        except ValueError:
            # not quick-mode, fall through to normal arg parsing
            pass

    parser = argparse.ArgumentParser(description='Hash table experiment runner')
    subparsers = parser.add_subparsers(dest='command')

    # generate_csv subcommand
    gen = subparsers.add_parser('generate_csv', help='Generate CSV over TABLE_SIZES')
    gen.add_argument('hash', type=int, choices=[1, 2, 3, 4], help='Hash function to use (1-4)')
    gen.add_argument('chunk', type=int, choices=[1, 2], help='Chunk to use: 1 for words 1-5000, 2 for words 5001-10000')
    gen.add_argument('--out', type=str, default='hash_results.csv', help='Output CSV path')

    # run subcommand (single table size run)
    run = subparsers.add_parser('run', help='Run a single table size experiment')
    run.add_argument('hash', type=int, choices=[1, 2, 3, 4])
    run.add_argument('chunk', type=int, choices=[1, 2])
    run.add_argument('--table_size', type=int, default=TABLE_SIZES[0])

    # compare subcommand: run all hashes at a single table size and emit per-index CSV
    comp = subparsers.add_parser('compare', help='Compare all hash functions at one table size')
    comp.add_argument('table_size', type=int, help='Table size to test')
    comp.add_argument('chunk', type=int, choices=[1, 2], help='Chunk to use: 1 for words 1-5000, 2 for words 5001-10000')
    comp.add_argument('--out', type=str, default='compare_hashes.csv', help='Output CSV path')

    args = parser.parse_args()

    if args.command == 'generate_csv':
        out = generate_csv(args.hash, args.chunk, out_path=args.out)
        print(f'Wrote CSV to: {out}')
    elif args.command == 'run':
        words_slice = _get_words_for_chunk(args.chunk)
        ht = HashTable(args.table_size)
        for w in words_slice:
            ht.insert(args.hash, w)
        stats = ht.stats()
        try:
            sd = statistics.stdev(ht.counts) if len(ht.counts) > 1 else 0
        except Exception:
            sd = 0
        stats['stddev'] = sd
        print(ht)
        print('Run stats:', stats)
        # written by copilot to export existing data to csv
        # write the table (Index, Count) to CSV
        out_table_csv = f'table_hash{args.hash}_chunk{args.chunk}_size{args.table_size}.csv'
        with open(out_table_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Index', 'Count'])
            for i, c in enumerate(ht.counts):
                writer.writerow([i, c])
        print(f'Wrote table CSV to: {out_table_csv}')
    elif args.command == 'compare':
        out = compare_hashes(args.table_size, args.chunk, out_path=args.out)
        print(f'Wrote compare CSV to: {out}')
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
