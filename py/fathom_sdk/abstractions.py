

class ClusteringContext:
    def __init__(self, clst_ctx_id: str,  ct_subc: int, ct_clst: int, ct_mega: int):
        self.clst_ctx_id: str = clst_ctx_id 
        self.ct_subc: int = ct_subc
        self.ct_clst: int = ct_clst 
        self.ct_mega: int = ct_mega 
        assert self.ct_tick > self.ct_subc
        assert self.ct_subc > self.ct_clst
        assert self.ct_clst > self.ct_mega
