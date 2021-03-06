# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/02_inference.ipynb (unless otherwise specified).

__all__ = ['Inferencer', 'main']

# Cell
# export

from fastai.text.all import *
from .core import *
from tqdm import tqdm

# Cell
class Inferencer(object):
    def __init__(self, learn): store_attr()

    def predict(self, item):
        dl = self.learn.dls.test_dl([item], rm_type_tfms=None, num_workers=0)
        pred = self.learn.model(dl.one_batch()[0])[0].sigmoid()
        pred_cls = torch.stack(torch.where(pred > 0.5))[1]
        dec = self.learn.dls.multi_categorize.decodes(pred_cls)
        return dec

# Cell
def _predict(model_name, items):
    learn = load_learner(model_name)
    infer = Inferencer(learn)

    items = Path(items)
    if items.is_file(): items = [items]
    elif items.is_dir(): items = items.ls()
    else: raise TypeError(f'expected items to be a file_name or directory name but got {type(items)}')
    texts = [open(item).read() for item in items]

    preds = []

    for text in tqdm(texts):
        p = infer.predict(text)
        preds.append(';'.join(p))

    return pd.DataFrame({'items': items,
                         'preds': preds})

# Cell
@call_parse
def main(model_name:Param('The path to the pickled fastai model', str) ,
            items:Param('A single file to predict or a folder with files to predict', str)):
    if model_name in _model_names.keys(): model_name = _model_names[model_name]
    df = _predict(model_name, items)
    df.to_csv(f'prediction_{Path(model_name).name}.csv', index=False)
    print(df)