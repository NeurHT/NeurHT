import torch
import torch.nn.functional as F

def Rotation_Loss(model,unlabled_data):
    rot_data = torch.cat([unlabled_data,
                          torch.rot90(unlabled_data,1,[2,3]),
                          torch.rot90(unlabled_data,2,[2,3]),
                          torch.rot90(unlabled_data,3,[2,3])],dim=0)
    rot_label = torch.cat([torch.zeros(len(unlabled_data)),
                           torch.ones(len(unlabled_data)),
                           torch.ones(len(unlabled_data))*2,
                           torch.ones(len(unlabled_data))*3],dim=0).to(unlabled_data).long()

    rot_pred = model.rot_forward(rot_data)
    rot_loss = F.cross_entropy(rot_pred,rot_label)
    return rot_loss



    