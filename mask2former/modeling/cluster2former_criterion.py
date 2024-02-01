"""
Cluster2Former criterion.
"""

import torch
import torch.nn.functional as F

def ccl_loss(prediction: torch.Tensor,
            target: torch.Tensor, 
            pair: torch.Tensor,
            weights = None,
            eps: float=1e-6):
    """
    Compute the cosine similarity clustring loss
    Args:
        prediction : Tensor in (Pixels, Q) format. Contains the sampled pixel features 
            with Softmax applied.
        target : Tensor in (Pixels) format. Contains the cluster target labels.
        pair : Tensor in (Pair_num, 2) format. Contains the generated pixel pair 
            indices.
        weights : Tensor in (Pixels) format. Contains the calculated weights for each 
            pixel pair.
        eps (Float): for numeric stability
    Returns:
        output (Float): the calculated loss
    """

    # indexing and loss computation
    target1 = target[pair[:, 0]]
    target2 = target[pair[:, 1]]
    sm1 = prediction[pair[:, 0]]
    sm2 = prediction[pair[:, 1]]

    sm1 /= sm1.sum(-1, keepdim=True) + eps
    sm2 /= sm2.sum(-1, keepdim=True) + eps

    R = torch.eq(target1, target2).float()
    
    P = (sm1 * sm2).sum(-1)

    L = (R * P + (1 - R) * (1 - P)) + eps
    L = -L.log() 
    
    if weights.shape[0] != 0:
        L *= weights
    
    return L.mean(0)

ccl_loss_jit = torch.jit.script(
    ccl_loss
)  # type: torch.jit.ScriptModule

def cos_sim_clustring_loss(prediction: torch.Tensor,
            target: torch.Tensor, 
            pair: torch.Tensor,
            weights = None,
            eps: float=1e-6):
    """
    Compute the cosine similarity clustring loss
    Args:
        prediction : Tensor in (Pixels, Q) format. Contains the sampled pixel features 
            with Softmax applied.
        target : Tensor in (Pixels) format. Contains the cluster target labels.
        pair : Tensor in (Pair_num, 2) format. Contains the generated pixel pair 
            indices.
        weights : Tensor in (Pixels) format. Contains the calculated weights for each 
            pixel pair.
        eps (Float): for numeric stability
    Returns:
        output (Float): the calculated loss
    """

    # indexing and loss computation
    target1 = target[pair[:, 0]]
    target2 = target[pair[:, 1]]
    sm1 = prediction[pair[:, 0]]
    sm2 = prediction[pair[:, 1]]

    cos_sim = F.cosine_similarity(sm1, sm2, dim=1, eps=1e-8)

    R = torch.eq(target1, target2).float()

    D = ((1 - cos_sim) + eps)/2

    L = -R * (1 - D).log() - (1 - R) * D.log()

    if weights.shape[0] != 0:
        L *= weights

    return L.mean(0)

cos_sim_clustering_loss_jit = torch.jit.script(
    cos_sim_clustring_loss
)  # type: torch.jit.ScriptModule
        
##########################################
### TODO: implement criterion
##########################################