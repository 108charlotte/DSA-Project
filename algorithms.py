def to_index(hashed_val, len): 
    return hashed_val % len

def hash_1(input): 
    return 0

def hash_2(input): 
    return 0

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
        self.entries = 0
        self.max_in_slot = -float('inf')
        self.min_in_slot = float('inf')
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
        index = to_index(self._hash(num, key), self.capacity)
        if self.table[index] is None: 
            self.table[index] = Node(hashed, value)
            self.size += 1
        else:
            current = self.table[index]
            while current:
                if current.key == key:
                    current.value = value
                    return
                current = current.next
            new_node = Node(hashed, value)
            new_node.next = self.table[index]
            self.table[index] = new_node
            self.size += 1

        self.entries += 1
    
    # (add later, don't want this to interfere w/ run times) def resize(self, new_capacity): 
        # self.capacity = new_capacity

    def search(self, num, key):
        index = self._hash(num, key)

        current = self.table[index]
        while current:
            if current.key == key:
                return current.value
            current = current.next
        raise KeyError(key)

    def populate(self, num, text): 
        word = ""
        for i in range(len(text)): 
            if text[i] == ' ': 
                self._insert(num, word, word)
            if text[i].isalpha(): 
                word += text[i]
    
    def __str__(self):
        elements = []
        for i in range(self.capacity):
            current = self.table[i]
            while current:
                elements.append((current.key, current.value))
                current = current.next
        return str(elements)


hash_table = HashTable(50)
hash_table.populate(1, "hi hi nope")
print(hash_table)