import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Umm so me using
# Auxilary and Global supervisions from the paper for ASD-NET 😁🫦
# As of now test1 be like, first extract features then think on it 
# Afterwards don't forget to implement this -> first think on it then extraction then again think on it, commenting so that I won't forget ig😁
class SpeakSense(nn.Module):

    def __init__(self, dim_size, map_size, out_size):
        super().__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(dim_size,map_size),
            nn.ReLU(),
            nn.Linear(map_size, map_size),
            nn.ReLU(),
            nn.Linear(map_size, out_size)
        )

        self.relu = nn.ReLU(inplace=True)

        self.fc_map_size_a = nn.Linear(dim_size, map_size)
        self.fc_map_size_v = nn.Linear(dim_size, map_size)

        self.fc_final = nn.Linear(map_size*2, 2)
        self.fc_aux_a = nn.Linear(map_size, 2)
        self.fc_aux_v = nn.Linear(map_size, 2)

        

    def forward(self, x, y):

        stream_feats = torch.concat((x, y), 1)

        a = self.relu(self.fc_map_size_a(x))
        v = self.relu(self.fc_map_size_v(y))

        aux_a = self.fc_aux_a(a)
        aux_v = self.fc_aux_v(v)

        av = torch.cat((a,v), 1)

        out = self.fc_final(av)
        
        return out, aux_a, aux_v, stream_feats
    

def train(model, optimizer, criterion,  train_dataloader, num_epochs=100, print_every=10):

    for epoch in tqdm(range(0, num_epochs), "Epochs"):
        total_loss = 0
        for _, b in tqdm(enumerate(train_dataloader), "Steps"):

            optimizer.zero_grad()

            av_out, a_out, v_out, _ = model(b[0], b[1])

            loss_out = criterion(av_out, b[2])
            loss_a = criterion(a_out, b[3])
            loss_v = criterion(v_out, b[4])

            loss = loss_out + loss_a + loss_v
            total_loss += loss
            loss.backward()
            optimizer.step()

        
        if epoch%print_every == 0:
            print(f"Total_Loss: {total_loss}")



if __name__ == "__main__":

    model = SpeakSense(512, 1024, 4)


    x = torch.randn((1, 512))
    y = torch.randn((1, 512))

    av_out, a_out, v_out, _ = model(x, y)

    print(av_out, a_out, v_out)

    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
    criterion = nn.CrossEntropyLoss()

    data = torch.randint(0, 5, (10, 2)).to(torch.float32)

    dataset = TensorDataset(torch.randn(10, 512), torch.randn(10, 512), data, data, data)

    train_dataloader = DataLoader(dataset, 2)


    train(model, optimizer, criterion, train_dataloader)

    