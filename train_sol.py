import time
import torch
from torch.utils.data import DataLoader
from network import PointNetLoss

#################### Training ####################
def TrainModel(
    task, model, train_loader:DataLoader = None, valid_loader:DataLoader = None,
    num_epochs:int = 10, optimizer = None, scheduler = None,
    device = None, save = True
    ):
    
    assert train_loader is not None
    assert optimizer is not None
    assert device is not None

    history = []

    train_start = time.time()
    for epoch in range(num_epochs): 
        model.train()
        history_epoch = {'epoch': epoch, 'train_loss':[], 'accuracy': None}
        epoch_start = time.time()
        for i, data in enumerate(train_loader, 0):
            start = time.time() # added by SH

            pointcloud = data['pointcloud'].to(device).float()
            category   = data['category'].to(device)

            optimizer.zero_grad()
            output, mat_in, mat_feature = model(pointcloud.transpose(1,2))

            loss = PointNetLoss(output, category, mat_in, mat_feature)
            loss.backward()
            optimizer.step()

            history_epoch['train_loss'].append(loss.item())

            # print statistics
            if (i+1) % 25 == 0:
                train_loss = round(loss.item(), 3)
                epoch_print, batch_print = epoch + 1, i + 1
                time_fb = round(time.time()-start, 2)
                time_epoch = round(time.time()- epoch_start, 2)
                txt1 = f"Epoch: {epoch_print: <2} / {num_epochs: <2}, Batch: {batch_print: <3} / {len(train_loader)} >>"
                txt2 = f"Train loss: {train_loss: <5}."
                txt3 = f"Forward-backward path (s) = {time_fb: <4}. Epoch time (s) = {time_epoch}"
                print(txt1, txt2, txt3)        
        
        if scheduler is not None:
            scheduler.step()

        # validation (or test)
        model.eval()
        correct, total = 0, 0

        if valid_loader is not None:
            with torch.no_grad():
                for data in valid_loader:
                    # get pointcloud and category
                    pointcloud = data['pointcloud'].to(device).float()
                    category   = data['category'].to(device)
                    # run the model
                    output, _, _ = model(pointcloud.transpose(1,2))
                    _, predict_category = torch.max(output.data, 1)
                    # record the correct predicitons
                    if task == "Classification":
                        total += predict_category.size(0)
                    elif task == "Segmentation":
                        total += predict_category.size(0) * predict_category.size(1) 

                    correct += (predict_category == category).sum().item()

            accuracy = int((correct / total) * 100)
            print(f"Accuracy: {accuracy}%\n---------------")

            if accuracy >= 90: return 

            history_epoch['accuracy'] = accuracy
        
        history.append(history_epoch)


        # save the model
        if save:
            torch.save(model.state_dict(), "save_"+str(epoch)+".pth")
    
    # after the whole training, print total time
    runtime = time.time() - train_start
    print(f"--------------------\nTotal runtime: {round(runtime)} sec")

    return history