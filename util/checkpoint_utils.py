import torch
import os

ROOT_DIR = os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir)))
CHECKPOINT_DIR = ROOT_DIR + '/checkpoints'


def save_checkpoint(model, optimizer, epoch: int, options):
    if not os.path.exists(CHECKPOINT_DIR):
        os.mkdir(CHECKPOINT_DIR)

    path = __get_checkpoint_path(model, epoch, options)
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(checkpoint, path)

    print('Saved checkpoint for \'{}\' at epoch #{}'.format(model.name, epoch))


def restore_from_checkpoint(model, optimizer, epoch: int, options):
    path = __get_checkpoint_path(model, epoch, options)
    if torch.cuda.is_available():
        checkpoint = torch.load(path)
    else:
        checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    print('Restored checkpoint for \'{}\' at epoch #{}'.format(model.name, epoch))


def __get_checkpoint_path(model, epoch, options):
    return os.path.join(CHECKPOINT_DIR, '{}_epoch_{}_b{}_p{}.pt'.format(model.name, epoch, options.batch_size,
                                                                        options.dropout_p))