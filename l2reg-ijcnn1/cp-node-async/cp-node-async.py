import base64
import io
import json
import logging
import os
import sys
import time
import torch
import torchvision
from torch.nn import functional
from torch.utils.data import TensorDataset, DataLoader
import pika
import numpy as np


cuda = torch.cuda.is_available()
device = 'cuda' if cuda else 'cpu'
default_tensor_str = 'torch.cuda.FloatTensor' if cuda else 'torch.FloatTensor'
torch.set_default_tensor_type(default_tensor_str)


def ecd(tensor) -> str:
    with io.BytesIO() as buf:
        torch.save(tensor.clone().detach(), buf)
        buf.seek(0)
        return base64.b85encode(buf.getvalue()).decode('utf-8')


def dcd(tensor: str) -> torch.Tensor:
    tensor = base64.b85decode(tensor.encode('utf-8'))
    with io.BytesIO(tensor) as buf:
        t = torch.load(buf, map_location=device).requires_grad_(False)
        return t


def pack_tensors(sender, **tensors) -> str:
    payload = {'sender': sender}
    for n, t in tensors.items():
        payload[n] = ecd(t) if t is not None else None
    return json.dumps(payload)


def unpack_tensors(payload, *tname):
    res = []
    payload = json.loads(payload)
    for n in tname:
        if payload[n] is None:
            res.append(None)
        else:
            res.append(dcd(payload[n]))
    return res


def comm_delay():
    DELAY = float(os.environ['DELAY'])
    time.sleep(np.random.lognormal(np.log(DELAY), 1.0) / 1000.0)


def loss_f(imgs, lbls, w):
    o = torch.matmul(imgs, w)
    return functional.cross_entropy(o, lbls)


ones_dxc = torch.ones((22, 2), device=device)
def loss_g(imgs, lbls, w, hp):
    l = loss_f(imgs, lbls, w)
    l = l + 0.5 * ((w ** 2) * torch.mul(hp.unsqueeze(1), ones_dxc)).mean()
    return l


def prep_data(sample_start, sample_end, batch_size, val_ratio):
    train_imgs = torch.load('./data/ijcnn1/X.tensor', map_location=device)[sample_start:sample_end]
    train_lbls = torch.load('./data/ijcnn1/y.tensor', map_location=device)[sample_start:sample_end]
    val_end = int((sample_end-sample_start)*val_ratio)
    train_imgs, val_imgs = train_imgs[:val_end], train_imgs[val_end:]
    train_lbls, val_lbls = train_lbls[:val_end], train_lbls[val_end:]
    logging.debug(f'train imgs shape: {train_imgs.shape}, train lbls shape: {train_lbls.shape}')
    logging.debug(f'val imgs shape: {val_imgs.shape}, val lbls shape: {val_lbls.shape}')
    train_set = DataLoader(TensorDataset(train_imgs, train_lbls), batch_size=batch_size)
    train_imgs_list, train_lbls_list = [], []
    for imgs, lbls in train_set:
        train_imgs_list.append(imgs)
        train_lbls_list.append(lbls)
    val_set = DataLoader(TensorDataset(val_imgs, val_lbls), batch_size=batch_size)
    val_imgs_list, val_lbls_list = [], []
    for imgs, lbls in val_set:
        val_imgs_list.append(imgs)
        val_lbls_list.append(lbls)
    return train_imgs_list, train_lbls_list, val_imgs_list, val_lbls_list


def main():
    logging.getLogger('pika.adapters.utils.io_services_utils').setLevel(logging.CRITICAL)
    logging.getLogger('pika.adapters.utils.connection_workflow').setLevel(logging.CRITICAL)
    logger = logging.getLogger('node-log')
    logger.setLevel(os.environ['log_level'])

    seed = int(os.environ['seed'])
    torch.manual_seed(seed)
    np.random.seed(seed)

    # consts
    host = sys.argv[1]
    sample_start = int(os.environ['SAMPLE_START'])
    sample_end = int(os.environ['SAMPLE_END'])
    if bool(int(os.environ['stochastic'])):
        batch_size = int(os.environ['batch_size'])
    else:
        batch_size = sample_end - sample_start  # deterministic
    w_lr = float(os.environ['w_lr'])
    hp_lr = float(os.environ['hp_lr'])
    val_ratio = float(os.environ['val_ratio'])
    mu = float(os.environ['mu'])
    n_features = 22
    n_classes = 2

    # data
    train_imgs_list, train_lbls_list, val_imgs_list, val_lbls_list = prep_data(sample_start, sample_end, batch_size, val_ratio)
    batch_size = train_imgs_list[0].shape[0]
    train_batch_num = len(train_imgs_list)
    val_batch_num = len(val_imgs_list)
    logger.debug(f'sample_start={sample_start}, sample_end={sample_end}, w_lr={w_lr}, hp_lr={hp_lr}, batch_size={batch_size}')
    logger.debug(f'train_batch_num={train_batch_num}, val_batch_num={val_batch_num}, val_ratio={val_ratio}')

    # params
    param = torch.zeros((n_features, n_classes), device=device, requires_grad=True)
    torch.manual_seed(seed)
    torch.nn.init.kaiming_normal_(param, mode='fan_in')
    hparam = torch.zeros((n_features,), device=device, requires_grad=True)
    torch.nn.init.normal_(hparam, 0, 1)
    b = None
    wk_approx = None
    v = None
    l = None

    def step_outer(lbda, theta):
        try:
            nonlocal param, wk_approx, hparam, l
            idx = np.random.randint(val_batch_num)
            # param
            l = loss_f(val_imgs_list[idx], val_lbls_list[idx], param)
            grad_w = torch.autograd.grad(l, param)[0]
            if lbda is not None:
                lbda = lbda.reshape([lbda.shape[0], 1, 1])
                grad_w = grad_w + torch.mul(lbda, b).sum(dim=0)
            param = param - w_lr * grad_w
            # hparam not dependent on Gi(y) in this problem
            if theta is not None:
                hparam = hparam - hp_lr * theta
            wk_approx = None
        except Exception:
            logger.exception('nd-step-w')

    def step_newcp(phi, zk):
        try:
            nonlocal wk_approx
            if wk_approx is None:
                nonlocal param
                wk_approx = param.clone().detach().requires_grad_()
            idx = np.random.randint(train_batch_num)
            l = loss_g(train_imgs_list[idx], train_lbls_list[idx], wk_approx, v)
            if phi is not None:
                l = l + torch.mul(phi, wk_approx).sum()
            l = l + 0.5 * mu * (wk_approx - zk).pow(2).sum()
            grad_wk = torch.autograd.grad(l, wk_approx, create_graph=True)[0]
            wk_approx = wk_approx - w_lr * grad_wk
        except Exception:
            logger.exception('nd-step-phix')

    def cb_recv(ch, method, props, body):
        try:
            nonlocal v, b
            msg_mode = props.headers['mode']
            if msg_mode == 'step_outer':
                # outer update
                lbda, theta = unpack_tensors(body, 'lbda', 'theta')
                step_outer(lbda, theta)
                payload = pack_tensors(host, x=hparam, y=param, fi=l)
                comm_delay()
                ch.basic_publish('', 'main', body=payload)
            elif msg_mode == 'step_newcp':
                # newcp update
                phi, zk = unpack_tensors(body, 'phi', 'zk')
                step_newcp(phi, zk)
                payload = pack_tensors(host, y=wk_approx)
                comm_delay()
                ch.basic_publish('', 'main', body=payload)
            elif msg_mode == 'grad_v':
                yt = unpack_tensors(body, 'yt')[0]
                h2 = torch.norm(yt - wk_approx, 2).pow(2)
                gradv = torch.autograd.grad(h2, v)[0]
                payload = pack_tensors(host, gradv=gradv)
                comm_delay()
                ch.basic_publish('', 'main', body=payload)
            elif msg_mode == 'update_v':
                v = unpack_tensors(body, 'v')[0]
                v.requires_grad_()
            elif msg_mode == 'update_b':
                b = unpack_tensors(body, 'b')[0]
        except Exception:
            logger.exception('nd-cb-recv')
        finally:
            ch.basic_ack(method.delivery_tag)

    
    # messaging
    connection = pika.BlockingConnection(pika.ConnectionParameters('myrabbitmq', retry_delay=1.0, connection_attempts=1000))
    channel = connection.channel()
    channel.queue_declare(host)
    channel.queue_declare('main')
    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(host, cb_recv)

    channel.start_consuming()


def crash_track(main):
    try:
        main()
    except Exception:
        logging.exception("node-error")


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout)
    crash_track(main)
