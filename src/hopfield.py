import numpy as np
import matplotlib.pyplot as plt
from dataloader import Dataloader

class Hopfield:
    def __init__(self,train_data_path,input_dim1,input_dim2):
        self.input_dim1 = input_dim1 + 1
        self.input_dim2 = input_dim2
        self.path = train_data_path[0]
        self.w = []
        self.threshold = None
        self.energy = 0
        self.energy_history = []
        with open(self.path,'r') as filecount:
            count = 0
            for line in filecount:
                count += 1
        with open(self.path,'r') as file:
            input_category = (count+1)//self.input_dim1
            curr_input_index = 0
            self.train_data = {i:[] for i in range(input_category)}
            for idx, line in enumerate(file):
                idx+=1
                if idx//self.input_dim1 == curr_input_index:
                    self.train_data[curr_input_index].append(line.replace(" ","0").strip())
                if idx%self.input_dim1 == 0:
                        curr_input_index+=1
        self.train_array = []
        for item in self.train_data.values():
            item = "".join(item)
            bipolar = []
            for i in range(len(item)):
                if item[i] == "0":
                    bipolar.insert(i,-1)
                else:
                    bipolar.insert(i,1)
            bipolar = np.array(bipolar)
            self.train_array.append(bipolar)
        self.train_array = np.array(self.train_array)
        self.categories = self.train_array.shape[0]
        self.flatten_input_dim = self.train_array.shape[1]
        self.state = np.random.randint(-1, 2, (self.flatten_input_dim,1))
    def train_hopfield(self):
        self.w = 1/self.flatten_input_dim*self.train_array.T@self.train_array #1/self.categories
        self.threshold = np.sum(self.w,axis=1)
        self.threshold = self.threshold.reshape(len(self.threshold),1)
        np.fill_diagonal(self.w,0)

    def update(self,enable_threshold,enable_async):
        if enable_async:
            for idx,row in enumerate(self.w):
                output_e = np.dot(row,self.state)
                output_e = output_e[0]
                if enable_threshold:
                    output_e -= self.threshold[idx]
                if  output_e > 0:
                    output_e = 1
                elif  output_e < 0:
                    output_e = -1
                self.state[idx] = output_e
        else:
            output = self.w @ self.state
            if enable_threshold:
                output = output - self.threshold
                output = output.reshape(len(output), 1)
            for i in range(len(output)):
                if output[i] > 0:
                    output[i] = 1
                elif output[i] < 0:
                    output[i] = -1
            self.state = output
    def input_state(self,input_vec):
        self.state = input_vec.reshape(len(input_vec),1)
    def display(self,mode):
        reshaped = self.state.reshape(self.input_dim1-1, self.input_dim2)
        print(reshaped)
        plt.imshow(reshaped, interpolation='nearest', cmap='gray_r')
        savefig_name = str(mode)+'.png'
        plt.savefig(savefig_name)
    def test(self,input_vec,enable_threshold,enable_async):
        self.input_state(input_vec)
        self.display(0)
        prev_energy = self.energy
        while True:
            self.update(enable_threshold,enable_async)
            curr_energy = self.compute_energy()
            if curr_energy == prev_energy:
                break
            prev_energy = curr_energy
        self.display(1)
    def compute_energy(self):
        self.energy = -0.5*np.dot(np.dot(self.state.T,self.w),self.state)
        self.energy_history.append(self.energy)
        return self.energy









# if __name__ == '__main__':
#     h = Hopfield("D:\Senior\\NNs\HWs\HW3\Hopfield_dataset\Bonus_Training.txt",10,10)
#     h.train_hopfield()
#     t = Dataloader("D:\Senior\\NNs\HWs\HW3\Hopfield_dataset\Bonus_Training.txt","D:\Senior\\NNs\HWs\HW3\Hopfield_dataset\Bonus_Testing.txt",10,10)
#     # test = t.get_test_sample(1)
#     test = t.gen_noisy_from_clean(0,0.02)
#     h.test(test)
#     h.display()

