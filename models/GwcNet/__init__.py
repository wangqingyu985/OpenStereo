from models.GwcNet.gwcnet import GwcNet_G, GwcNet_GC
from models.GwcNet.loss import model_loss

__models__ = {
    "gwcnet-g": GwcNet_G,
    "gwcnet-gc": GwcNet_GC
}
