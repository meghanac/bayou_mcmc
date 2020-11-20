import argparse
import pickle
from cryptography.fernet import Fernet


class SecretSanta:
    def __init__(self, key, assignment):
        self.key = key
        self.encryption = assignment

    def print_assignment(self):
        self.key = Fernet(self.key)
        print(self.key.decrypt(self.encryption))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('your_name', type=str,
                        help='Enter your name, example command: python secret_santa_reader.py meghana')
    clargs = parser.parse_args()
    clargs.your_name.capitalize()
    f = open(clargs.your_name + "_assignment.pickle", "rb")
    pickle.load(f).print_assignment()
    f.close()