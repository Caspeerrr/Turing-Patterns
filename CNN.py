import torch
import numpy as np

from models import FCN, save_model, CNNClassifier
import torch.utils.tensorboard as tb
import torchvision.datasets as datasets
import torchvision
import torchvision.transforms as transforms

    
def train(train_data,
          valid_data,
          log_dir, 
          num_epoch = 20,
          lr = 1e-3,
          gamma = 0,
          continue_training = False,
          transform= "Compose([RandomHorizontalFlip(), ToTensor()])"
         ):

         # ColorJitter -> Randomly change the brightness, contrast, saturation and hue of an image
         # RandomHorizontalFlip ->Horizontally flip the given image randomly with a given probability
    
    model = CNNClassifier(layers=[16, 32, 64, 128], n_output_channels=1, kernel_size=3)
    train_logger, valid_logger = None, None

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = CNNClassifier().to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss = torch.nn.MSELoss()

    # import inspect
    # transform = eval(transform, {k: v for k, v in inspect.getmembers(torchvision.transforms) if inspect.isclass(v)})

    global_step = 0
    for epoch in range(num_epoch):
        model.train()
        print('Epoch: ', epoch)

        for img, label in train_data:
            img = img.to(device)
            logit = model(img)
            loss_val = loss(logit[0], label)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            global_step += 1

        model.eval()
        count = 0
        total_mse = 0
        lossmae = torch.nn.L1Loss(reduction='none')
        for img, label in valid_data:
            img, label = img.to(device), label.to(device).long()
            logit = model(img)

            total_mse += lossmae(logit[0], label)
            count += 1

        total_mse /= count
        print(total_mse)

        if valid_logger is None or train_logger is None:
            print('mae = ' + str(total_mse))
        save_model(model)
