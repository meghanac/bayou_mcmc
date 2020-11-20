import pickle
import os
import random
from cryptography.fernet import Fernet

names = ["Meghana", "Justin", "Erik", "Harrison"]

# Brute force
# shuffled = names.copy()
# random.shuffle(shuffled)
# while True:
#     if sum([int(names[i] != shuffled[i]) for i in range(len(names))]) == 4:
#         break
#     random.shuffle(shuffled)

random.shuffle(names)

key = Fernet.generate_key()
cipher_suite = Fernet(key)


class SecretSanta:
    def __init__(self, key, assignment):
        self.key = key
        # self.encryption = cipher_suite.encrypt(bytes(assignment, 'utf-8'))
        self.encryption = assignment

    def print_assignment(self):
        self.key = Fernet(self.key)
        print(self.key.decrypt(self.encryption))


for i in range(len(names)):
    # Brute force
    # assignment = shuffled[i]

    # Smarter method
    assignment = names[(i + 1) % len(names)]

    # print(names[i], assignment)

    f = open(names[i] + "_assignment.pickle", "wb")
    pickle.dump(SecretSanta(key, cipher_suite.encrypt(bytes(assignment, 'utf-8'))), f)
    f.close()

# for i in range(len(names)):
#     pickle.load(open(names[i] + "_assignment.pickle", "rb")).print_assignment()
