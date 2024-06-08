import torch
from numpy.random import default_rng
from .vessal import StreamingSampling

# WIP
class StreamingRandTorch(StreamingSampling):
    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super(StreamingRandTorch, self).__init__(X, Y, idxs_lb, net, handler, args)
        self.skipped = []
        self.rng = default_rng()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.debug = True


    def query(self, n):#

        # Use allowed range if exists, if defined, else use fallback
        if hasattr(self, "allowed"):
            cand : torch.Tensor = self.allowed
        else:
            cand = self.get_valid_candidates()

        # map n to device if it is not yet
        n = torch.tensor(n, device=self.device, dtype=torch.int) if not isinstance(n, torch.Tensor) else n
        # create indices
        idxs_cand = torch.nonzero(cand).squeeze()
        # select randomly
        idxs_cand = idxs_cand[torch.randperm(idxs_cand.shape[0], device=self.device)[:min(n, idxs_cand.shape[0])]]

        if self.debug:
            print('chosen: {}, skipped: {}, n:{}'.format(len(idxs_cand), cand.sum(), n))

        return idxs_cand
