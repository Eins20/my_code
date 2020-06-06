from by_dialate import images_gen
from app import open_flask

def edge_show(true_image_dir,pred_image_dir):
    images_gen(true_image_dir, pred_image_dir)
    open_flask()

if __name__ == '__main__':
    edge_show("./ref","./our_pred")