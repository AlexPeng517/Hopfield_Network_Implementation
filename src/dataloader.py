import numpy as np
class Dataloader:
    def __init__(self,train_data_path,test_data_path,input_dim1,input_dim2):
        self.input_dim1 = input_dim1 +1
        self.input_dim2 = input_dim2
        self.test_path = test_data_path[0]
        self.train_path = train_data_path[0]
        self.test_data_arr = []
        self.train_data_arr = []
        with open(self.test_path, 'r') as test_filecount:
            test_count = 0
            for line in test_filecount:
                test_count += 1
        with open(self.train_path, 'r') as train_filecount:
            train_count = 0
            for line in train_filecount:
                train_count += 1
        with open(self.test_path, 'r') as test_file:
            self.test_entries_num = (test_count + 1) // self.input_dim1
            curr_input_index = 0
            self.test_data = {i: [] for i in range(self.test_entries_num)}
            for idx, line in enumerate(test_file):
                idx += 1
                if idx // self.input_dim1 == curr_input_index:
                    self.test_data[curr_input_index].append(line.replace(" ", "0").strip())
                if idx % self.input_dim1 == 0:
                    curr_input_index += 1
        self.test_data_array = []
        for item in self.test_data.values():
            item = "".join(item)
            bipolar = []
            for i in range(len(item)):
                if item[i] == "0":
                    bipolar.insert(i, -1)
                else:
                    bipolar.insert(i, 1)
            bipolar = np.array(bipolar)
            self.test_data_array.append(bipolar)

        with open(self.train_path, 'r') as train_file:
            self.train_entries_num = (train_count + 1) // self.input_dim1
            curr_input_index = 0
            self.train_data = {i: [] for i in range(self.train_entries_num)}
            for idx, line in enumerate(train_file):
                idx += 1
                if idx // self.input_dim1 == curr_input_index:
                    self.train_data[curr_input_index].append(line.replace(" ", "0").strip())
                if idx % self.input_dim1 == 0:
                    curr_input_index += 1
        self.train_data_array = []
        for item in self.train_data.values():
            item = "".join(item)
            bipolar = []
            for i in range(len(item)):
                if item[i] == "0":
                    bipolar.insert(i, -1)
                else:
                    bipolar.insert(i, 1)
            bipolar = np.array(bipolar)
            self.train_data_array.append(bipolar)

    def get_test_sample(self,index):
        return self.test_data_array[index].reshape(len(self.test_data_array[index]),1).copy()

    def gen_noisy_from_clean(self,ref,degree):
        clean = self.train_data_array[ref]
        noisy = clean.copy()
        num_elements_to_modify = int(len(clean)*degree)
        for i in range(num_elements_to_modify):
            rand_index = np.random.randint(0,len(clean))
            noisy[rand_index]*=-1
        return noisy.copy()



