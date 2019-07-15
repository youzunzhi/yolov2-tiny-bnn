from test_option import Options
from torch.utils.data import DataLoader
from dataset import Yolov2Dataset
from model import Model
if __name__ == "__main__":
    options = Options().options


    dataset = Yolov2Dataset(options)
    dataloader = DataLoader(
        dataset,
        batch_size=options.batch_size,
        shuffle=False,
        num_workers=1,
        collate_fn=dataset.collate_fn
    )

    model = Model(options)
    model.load_weights(options.weights_path)
    model.eval(dataloader)
