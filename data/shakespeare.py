import os
import tiktoken
import numpy as np
import requests


class PreprocessData(object):

    def __init__(
            self,
            url: str = None,
            encoding_str: str = "gpt2",
            train_size: int = 0.9,
    ):
        """
        Initialize a PreprocessData class.
        :param url: request the data from this url
        :param encoding_str: type of encoding we will be using for our tokenizer
        :param train_size: proportion of training data. Should be between 0 and 1.
        """
        assert train_size >= 0 or train_size <= 1

        self.url = url
        self.encoding_str = encoding_str
        self.train_size = train_size
        self.input_file_path = os.path.join(os.path.dirname(__file__), "input.txt")

        if self.url is None:
            # By default, we will be working with Shakespeare data.
            self.url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'

        # Request the data
        if not os.path.exists(self.input_file_path):
            with open(self.input_file_path, "w") as f:
                f.write(requests.get(self.url).text)

        # Make the data buffer
        with open(self.input_file_path, "r") as f:
            self.data = f.read()

        # Split the data into training and validation sets
        self.train_data = self.data[:int(self.data_len * self.train_size)]
        self.val_data = self.data[int(self.data_len * self.train_size):]

        # Make the encoding and store training and test data in two arrays.
        self.encoding = tiktoken.get_encoding(self.encoding_str)
        self.train_idx = np.array(self.encoding.encode_ordinary(self.train_data), dtype=np.uint16)
        self.val_idx = np.array(self.encoding.encode_ordinary(self.val_data), dtype=np.uint16)

    @property
    def data_len(self):
        """
        Access the length of the data
        :return: data length : int
        """
        return len(self.data)

    @property
    def get_data(self):
        """
        Access the data through a getter.
        :return: data : str
        """
        return self.data

    def train_val_to_file(self, train_file, val_file):

        """
        Save the tokens in two binary files so that we can access it later.

        :param train_file: name of the file where training tokens will be saved
        :param val_file: name of the file where val tokens will be saved
        :return: None
        """

        assert type(train_file) == str
        assert type(val_file) == str

        self.train_idx.tofile(os.path.join(os.path.dirname(__file__), train_file))
        self.val_idx.tofile(os.path.join(os.path.dirname(__file__), val_file))
