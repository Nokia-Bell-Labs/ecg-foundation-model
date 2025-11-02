import lightning as L
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


def z_score_normalize(tensor, dim=0, eps=1e-6):
    """Z-score normalization"""
    mean = tensor.mean(dim=dim, keepdim=True)
    std = tensor.std(dim=dim, keepdim=True)
    return (tensor - mean) / (std + eps)

def min_max_normalize(tensor, eps=1e-6):
    """Min-max normalization to [0, 1]"""
    min_val = tensor.min()
    max_val = tensor.max()
    return (tensor - min_val) / (max_val - min_val + eps)

def get_off_diagonal_min_max(matrix):
    """
    Returns the min and max values of a square matrix, excluding the diagonal.
    """
    assert matrix.dim() == 2 and matrix.size(0) == matrix.size(1), "Input must be a square matrix"
    mask = ~torch.eye(matrix.size(0), dtype=torch.bool, device=matrix.device)
    off_diag = matrix[mask]
    min_val = off_diag.min()
    max_val = off_diag.max()
    return min_val, max_val

class ContrastiveModel(L.LightningModule):
    def __init__(self, backbone: nn.Module, model_configs, exp_configs, **kwargs):
        super().__init__()
        self.backbone = backbone
        self.lr = exp_configs.learning_rate
        self.weight_decay = exp_configs.weight_decay
        self.temperature = exp_configs.temperature
        self.use_metadata = exp_configs.use_metadata

        self.pretrain_method = exp_configs.get('pretrain_method', 'simclr')  # default to simclr
        if self.backbone.__class__.__name__.lower() == "xresnet1d":
            self.feature_dim = 768
        else:
            self.feature_dim = self._get_feature_dim()
        
        
        if self.pretrain_method == 'moco':
            self._init_moco(exp_configs)
        elif self.pretrain_method == 'byol':
            self._init_byol(exp_configs)
        elif self.pretrain_method == 'simclr':
            pass  
        else:
            raise ValueError(f"Unknown pretrain_method: {self.pretrain_method}")

    ######### Init #########
    def _get_feature_dim(self):
        """Get the output feature dimension of the backbone"""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dummy_input = torch.randn(1, 1, 5000).to(device) 
        with torch.no_grad():
            if self.backbone.__class__.__name__.lower() == "st_mem_vit":
                features = self.backbone(dummy_input[:,:,:4950])
            else:
                features = self.backbone(dummy_input)
            if hasattr(features, "embeddings"):
                return features.embeddings.shape[-1]
            else:
                return features.shape[-1]

    def _init_moco(self, exp_configs):
        """Initialize MoCo-specific components"""
        self.moco_k = exp_configs.get('moco_k', 65536)  # Queue size
        self.moco_m = exp_configs.get('moco_m', 0.999)   # Momentum coefficient
        
        self.backbone_k = copy.deepcopy(self.backbone)
        
        for param_q, param_k in zip(self.backbone.parameters(), self.backbone_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        
        self.register_buffer("queue", torch.randn(self.feature_dim, self.moco_k))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def _init_byol(self, exp_configs):
        """Initialize BYOL-specific components"""
        self.byol_tau = exp_configs.get('byol_tau', 0.996)  # Target network update rate
        
        # Predictor network for BYOL
        self.predictor = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.BatchNorm1d(self.feature_dim // 2),
            nn.ReLU(),
            nn.Linear(self.feature_dim // 2, self.feature_dim)
        )
        
        self.backbone_target = copy.deepcopy(self.backbone)
        
        for param_online, param_target in zip(self.backbone.parameters(), self.backbone_target.parameters()):
            param_target.data.copy_(param_online.data)
            param_target.requires_grad = False

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Momentum update of the key encoder (MoCo)"""
        for param_q, param_k in zip(self.backbone.parameters(), self.backbone_k.parameters()):
            param_k.data = param_k.data * self.moco_m + param_q.data * (1. - self.moco_m)

    @torch.no_grad()
    def _update_target_network(self):
        """Update target network (BYOL)"""
        for param_online, param_target in zip(self.backbone.parameters(), self.backbone_target.parameters()):
            param_target.data = self.byol_tau * param_target.data + (1 - self.byol_tau) * param_online.data

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """Update the queue (MoCo)"""
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.moco_k % batch_size == 0
        
        if ptr + batch_size <= self.moco_k:
            self.queue[:, ptr:ptr + batch_size] = keys.T
        else:
            remaining_space = self.moco_k - ptr

            if remaining_space > 0:
                self.queue[:, ptr:] = keys[:remaining_space].T
            
            overflow = batch_size - remaining_space
            if overflow > 0:
                self.queue[:, :overflow] = keys[remaining_space:].T
            
        ptr = (ptr + batch_size) % self.moco_k  # Move pointer

        self.queue_ptr[0] = ptr
    

    ######### Foward #########
    def forward(self, x):
        if self.backbone.__class__.__name__.lower() == "st_mem_vit":
            features = self.backbone(x[:,:,:4950])
        else:
            features = self.backbone(x)
        if hasattr(features, "embeddings"):
            features = features.embeddings
        else:
            features = features # resnet
        return features

    def forward_target(self, x):
        """Forward through target/momentum encoder"""
        if self.pretrain_method == 'moco':
            features = self.backbone_k(x)
        elif self.pretrain_method == 'byol':
            features = self.backbone_target(x)
        else:
            raise ValueError(f"Target forward not supported for {self.pretrain_method}")
            
        if hasattr(features, "embeddings"):
            features = features.embeddings
        else:
            features = features
        return features

    ########## Similarity #########
    def _get_metadata_similarity(self, metadata):
        if metadata.ndim > 1:
            mask = metadata[:,0]
            metadata = metadata[:,-1]
            mask_flag = 1
        else:
            mask_flag = 0

        metadata = metadata.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
        risk_scores = metadata.unsqueeze(1)
        risk_diff = (risk_scores - risk_scores.T)**2

        risk_diff_norm = min_max_normalize(risk_diff)
        temp = 0.2 # for controlling how hard the samples are penalised 
        # lower -- more seperated; higher -- the weight differences between the negatives are less
        risk_diff_norm = (1 - temp) * risk_diff_norm + temp
        risk_diff_norm.fill_diagonal_(0.0)

        # add missing metadata control
        if mask_flag:
            mask_metrix = torch.exp(- mask.unsqueeze(1) * mask.unsqueeze(0))
            risk_diff_norm = risk_diff_norm * mask_metrix
                
        meta_similarity = 1 - risk_diff_norm
        return meta_similarity
    
    def _get_feature_similarity(self, features):
        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T)
        return similarity_matrix

    ######### Loss #########
    def _info_nce_loss(self, features):
        batch_size = features.size(0) // 2

        labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(features.device)

        similarity_matrix = self._get_feature_similarity(features)

        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(features.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(features.device)

        logits = logits / self.temperature
        loss = F.cross_entropy(logits, labels) # exp is here
        return loss
    
    def _similarity_alignment_loss(self, features, metadata):
        batch_size = features.size(0) // 2
        meta_score = self._get_metadata_similarity(metadata)

        # expand to match the shape of 2*batch * 2*batch size
        meta_sim_matrix = torch.zeros(2*batch_size, 2*batch_size, device=features.device)
        meta_sim_matrix[:batch_size, :batch_size] = meta_score
        meta_sim_matrix[:batch_size, batch_size:] = meta_score
        meta_sim_matrix[batch_size:, :batch_size] = meta_score
        meta_sim_matrix[batch_size:, batch_size:] = meta_score

        feature_similarity = self._get_feature_similarity(features)
        similarity_norm = min_max_normalize(feature_similarity)
        # Alignment loss: minimize the difference between feature and metadata similarity
        alignment_loss = F.mse_loss(similarity_norm, meta_sim_matrix)

        return alignment_loss

    def _meta_nce_loss(self, features, metadata):
        batch_size = features.size(0) // 2
        meta_score = self._get_metadata_similarity(metadata)

        meta_sim_matrix = torch.zeros(2*batch_size, 2*batch_size, device=features.device)
        meta_sim_matrix[:batch_size, :batch_size] = meta_score
        meta_sim_matrix[:batch_size, batch_size:] = meta_score
        meta_sim_matrix[batch_size:, :batch_size] = meta_score
        meta_sim_matrix[batch_size:, batch_size:] = meta_score
        
        labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(features.device)
        
        similarity_matrix = self._get_feature_similarity(features)
        scaled_similarity = similarity_matrix / self.temperature # scaling with temperature
        weights = 1 - meta_sim_matrix # reverse the weight -- higher meta similarity, lower weight
        weights = F.softmax(weights, dim=1) # add

        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(features.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = scaled_similarity[~mask].view(scaled_similarity.shape[0], -1)
        weights = weights[~mask].view(weights.shape[0], -1)
        
        similarity_matrix = similarity_matrix - similarity_matrix.max(dim=1, keepdim=True).values
        exp_similarities = torch.exp(similarity_matrix)
        weighted_exp_similarities = weights * exp_similarities
        
        positives = exp_similarities[labels.bool()].view(labels.shape[0], -1) # numerator
        negatives = weighted_exp_similarities[~labels.bool()].view(weighted_exp_similarities.shape[0], -1)
        
        positive_sum = positives.sum(dim=1)
        negative_sum = negatives.sum(dim=1)
        total_sum = positive_sum + negative_sum
        
        log_probs = torch.log(positive_sum / (total_sum + 1e-8))
        loss = -log_probs.mean()
        
        return loss

    def _moco_loss(self, q, k):
        """MoCo contrastive loss"""
        # Normalize features
        q = F.normalize(q, dim=1)
        k = F.normalize(k, dim=1)
        
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)
        
        self._dequeue_and_enqueue(k)
        loss = F.cross_entropy(logits, labels)
        return loss

    def _byol_loss(self, p1, z2, p2, z1):
        """BYOL loss (symmetric)"""
        def cosine_loss(p, z):
            p = F.normalize(p, dim=1)
            z = F.normalize(z, dim=1)
            return 2 - 2 * (p * z).sum(dim=1)
        
        loss1 = cosine_loss(p1, z2)
        loss2 = cosine_loss(p2, z1)
        loss = (loss1 + loss2).mean()
        return loss
    
    ########## Training Step #########
    def training_step(self, batch, batch_idx):
        use_mask = False
        signals, label, ch_name, ch_freq, metadata = batch
        signals = signals.transpose(0, 1)

        x1, x2 = signals[0], signals[1]
        
        if self.pretrain_method == 'simclr':
            x = torch.cat([x1, x2], dim=0)
            z = self.forward(x.unsqueeze(1))
            if self.backbone.__class__.__name__.lower() == "xresnet1d":
                z = torch.mean(z, dim=-1)
            
            if self.use_metadata:
                if use_mask:
                    metadata = metadata[:,-2:]
                else:
                    metadata = metadata[:,-1]
                loss_nce = self._meta_nce_loss(z, metadata)
                loss_sim = self._similarity_alignment_loss(z, metadata)
                loss = (loss_nce + loss_sim) / 2
            else: 
                loss = self._info_nce_loss(z)
                
        elif self.pretrain_method == 'moco':
            q = self.forward(x1.unsqueeze(1))
            with torch.no_grad():
                self._momentum_update_key_encoder()

                k = self.forward_target(x2.unsqueeze(1))
            
            loss = self._moco_loss(q, k)
            self._dequeue_and_enqueue(k)
            
        elif self.pretrain_method == 'byol':
            z1 = self.forward(x1.unsqueeze(1))
            z2 = self.forward(x2.unsqueeze(1))
            
            p1 = self.predictor(z1)
            p2 = self.predictor(z2)
            
            with torch.no_grad():
                self._update_target_network()
                z1_target = self.forward_target(x1.unsqueeze(1))
                z2_target = self.forward_target(x2.unsqueeze(1))
            
            loss = self._byol_loss(p1, z2_target, p2, z1_target)
        
        else:
            raise ValueError(f"Unknown pretrain_method: {self.pretrain_method}")
    
        self.log("train_loss", loss, on_step=True, on_epoch=True, logger=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999)
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,  
            T_mult=2, 
            eta_min=1e-6 
        )
        
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}



