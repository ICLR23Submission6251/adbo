import argparse
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
import pika
import numpy as np
import pandas as pd


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


def propWithModeHeader(str_mode: str):
    return pika.BasicProperties(headers={"mode": str_mode})
    

def test_avg_loss(imgs, lbls, w):
    o = torch.matmul(imgs, w)
    l = functional.cross_entropy(o, lbls)
    acc = (o.argmax(1) == lbls).sum() / lbls.shape[0]
    return l, acc


def get_c1(t: int) -> float:
    # constant impl
    return float(os.environ['c1'])
    

def get_c2(t: int) -> float:
    # constant impl
    return float(os.environ['c2'])


def save(exp_time, eph, csvlog: pd.DataFrame):
    logfolder = f'./log/{exp_time}/'
    if not os.path.exists(logfolder):
        os.makedirs(logfolder)
        # exp meta data: params
        with open(logfolder + 'setting.txt', 'w') as fmeta:
            EXP_ENV_KEYS = [
                'seed', 
                'train_size', 'w_lr', 'hp_lr', 'batch_size',
                'stochastic', 'val_ratio',
                'z_lr', 'v_lr', 'lbda_lr', 'lbda_init',
                'phi_lr', 'theta_lr', 'c1', 'c2', 'mu',
                'pre_epochs', 'phix_iter',
                'active_set', 'S', 'tau'
            ]
            fmeta.write(f'exp: ijcnn1_l2reg_adbo\n')
            for k in EXP_ENV_KEYS:
                fmeta.write(f'{k}: {os.environ[k]}\n')
            fmeta.write(f'--------\n')
    np.savetxt(logfolder + f'eph_{eph}.txt', csvlog.values, fmt='%9.7f  %6.4f  %f  %d')
    # csvlog.to_csv(logfolder + f'eph_{eph}.csv', index=False)


def main():
    logging.getLogger('pika.adapters.utils.io_services_utils').setLevel(logging.CRITICAL)
    logging.getLogger('pika.adapters.utils.connection_workflow').setLevel(logging.CRITICAL)
    # parse arg to read node hosts
    parser = argparse.ArgumentParser()
    parser.add_argument('node_hosts', nargs='+')
    args = parser.parse_args()
    NODES = args.node_hosts
    NODES_CNT = len(NODES)
    
    # parse env
    logger = logging.getLogger('main-log')
    logger.setLevel(os.environ['log_level'])
    logger.debug(args)

    seed = int(os.environ['seed'])
    torch.manual_seed(seed)
    np.random.seed(seed)

    v_lr = float(os.environ['v_lr'])
    z_lr = float(os.environ['z_lr'])
    lbda_lr = float(os.environ['lbda_lr'])
    theta_lr = float(os.environ['theta_lr'])
    phi_lr = float(os.environ['phi_lr'])
    epochs = int(os.environ['epochs'])
    pre_epochs = int(os.environ['pre_epochs'])
    S = int(os.environ['S'])
    tau = int(os.environ['tau'])
    active_set = bool(int(os.environ['active_set']))
    phix_iter = int(os.environ['phix_iter'])
    mu = float(os.environ['mu'])

    exp_time = int(time.time())

    train_size = int(os.environ['train_size'])
    t_imgs = torch.load('./data/ijcnn1/X.tensor', map_location=device)[train_size:]
    t_lbls = torch.load('./data/ijcnn1/y.tensor', map_location=device)[train_size:]
    logger.debug(f'test imgs shape: {t_imgs.shape}, test lbls shape: {t_lbls.shape}')
    
    # messaging
    connection = pika.BlockingConnection(pika.ConnectionParameters('myrabbitmq', retry_delay=1.0, connection_attempts=1000))
    channel = connection.channel()
    for nd in NODES:
        channel.queue_declare(nd)
    channel.queue_declare('main')
    main_consumer = channel.consume('main')

    n_features = 22
    n_classes = 2

    z = torch.zeros((n_features, n_classes), device=device)
    torch.manual_seed(seed)
    torch.nn.init.kaiming_normal_(z, mode='fan_in')
    ws = dict(zip(NODES, [z.clone().detach() for _ in range(NODES_CNT)]))
    v = torch.zeros((n_features,), device=device)
    torch.nn.init.normal_(v, 0, 1)
    hps = dict(zip(NODES, [v.clone().detach() for _ in range(NODES_CNT)]))
    thetas = dict(zip(NODES, [v.clone().detach() for _ in range(NODES_CNT)]))
    lbda = []
    pa = []
    pb = []
    pc = []
    pk = []
    taus = dict(zip(NODES, [1 for _ in range(NODES_CNT)]))

    loss_history = []
    train_log = pd.DataFrame(columns=['test_loss', 'test_acc', 'time', 'comm'])
    fi = {}
    start_loss, start_acc = test_avg_loss(t_imgs, t_lbls, z)
    logger.info(f'eph = {0:5d}, test_loss = {start_loss:6.4f}, test_acc = {100*start_acc:7.4f}')
    starttime = time.time()
    res = [start_loss.item(), start_acc.item(), 0.0, 0]
    train_log.loc[len(train_log)] = res
    loss_history.append(res)
    for nd in NODES:
        channel.basic_publish('', nd, body=pack_tensors('main', lbda=None, theta=thetas[nd]), properties=propWithModeHeader('step_outer'))
    for k in range(epochs):
        active_nds = []
        # recv from S 
        while True:
            mtd, props, body = next(main_consumer)
            # sender, x, y
            body = json.loads(body)
            hps[body['sender']] = dcd(body['x'])
            ws[body['sender']] = dcd(body['y'])
            fi[body['sender']] = dcd(body['fi'])
            active_nds.append(body['sender'])
            taus[body['sender']] = 0
            channel.basic_ack(mtd.delivery_tag)
            # check condition
            if len(active_nds) >= S and max(taus.values()) < tau:
                break
        for nd in NODES:
            taus[nd] += 1
        # logger.debug(f'eph = {(k+1):5d}, active: {active_nds}')
        # update v, z
        gradv = -torch.stack(list(thetas.values())).sum(dim=0)
        for l_l, a_l in zip(lbda, pa):
            gradv = gradv + l_l * a_l
        v = v - v_lr * gradv
        gradz = 0
        for l_l, c_l in zip(lbda, pc):
            gradz = gradz + l_l * c_l
        # if len(pc) > 0:
        #     logger.debug(f'gradz_norm: {torch.norm(gradz)}')
        z = z - z_lr * gradz
        # update lbda
        for i in range(len(lbda)):
            grad_lbdal = torch.mul(pa[i], v).sum()
            grad_lbdal = grad_lbdal + torch.mul(pc[i], z).sum()
            grad_lbdal = grad_lbdal + pk[i]
            for nd in NODES:
                grad_lbdal = grad_lbdal + torch.mul(ws[nd], pb[i][nd]).sum()
            grad_lbdal = grad_lbdal - get_c1(k+1) * lbda[i]
            lbda[i] = functional.relu(lbda[i] + lbda_lr * grad_lbdal)
        # update theta
        for nd in active_nds:
            grad_thetai = hps[nd] - v - get_c2(k+1) * thetas[nd]
            thetas[nd] = thetas[nd] + theta_lr * grad_thetai
        # send back to active nodes
        lbda_nd = torch.tensor(lbda) if len(lbda) > 0 else None
        for nd in active_nds:
            channel.basic_publish('', nd, body=pack_tensors('main', lbda=lbda_nd, theta=thetas[nd]), properties=propWithModeHeader('step_outer'))    

        if k % pre_epochs == 0:
            # active set
            if active_set:
                for j in reversed(range(len(lbda))):
                    if lbda[j] < 1e-10:
                        del lbda[j]
                        del pa[j]
                        del pb[j]
                        del pc[j]
                        del pk[j]
            
            h2z = z.clone().detach()
            h2y = {}
            phi = dict(zip(NODES, [torch.zeros_like(z) for _ in range(NODES_CNT)]))
            # obtain all worker wk sync
            for _ in range(NODES_CNT):
                mtd, props, body = next(main_consumer)
                # sender, x, y
                body = json.loads(body)
                h2y[body['sender']] = dcd(body['y'])
                channel.basic_ack(mtd.delivery_tag)
            for nd in NODES:
                taus[nd] = 1
                channel.basic_publish('', nd, body=pack_tensors('main', v=v), properties=propWithModeHeader('update_v'))
                channel.basic_publish('', nd, body=pack_tensors('main', phi=phi[nd], zk=h2z), properties=propWithModeHeader('step_newcp'))
            for _ in range(phix_iter):
                active_nds = []
                # recv from S 
                while True:
                    mtd, props, body = next(main_consumer)
                    # sender, y
                    body = json.loads(body)
                    h2y[body['sender']] = dcd(body['y'])
                    active_nds.append(body['sender'])
                    taus[body['sender']] = 0
                    channel.basic_ack(mtd.delivery_tag)
                    # check condition
                    # 20230131 UPDATE: DO IT SYNC (IF YOU WANT)
                    if len(active_nds) >= NODES_CNT and max(taus.values()) < 1:
                        break
                for nd in NODES:
                    taus[nd] += 1
                # update zk
                gradz = -torch.stack(list(phi.values())).sum(dim=0)
                # update phik
                for nd in active_nds:
                    grad_phik = h2y[nd] - h2z
                    gradz += mu * (-grad_phik)
                    phi[nd] = phi[nd] + phi_lr * grad_phik
                h2z = h2z - z_lr * gradz
                # send back
                for nd in active_nds:
                    channel.basic_publish('', nd, body=pack_tensors('main', phi=phi[nd], zk=h2z), properties=propWithModeHeader('step_newcp'))

            # TODO : drop N updates
            for _ in range(NODES_CNT):
                mtd, props, body = next(main_consumer)
                channel.basic_ack(mtd.delivery_tag)
            # get grad v sync
            for nd in NODES:
                channel.basic_publish('', nd, body=pack_tensors('main', yt=ws[nd]), properties=propWithModeHeader('grad_v'))
            # new cp
            gradv = 0
            for _ in range(NODES_CNT):
                mtd, props, body = next(main_consumer)
                # sender, gradv
                body = json.loads(body)
                gradv = gradv + dcd(body['gradv'])
                channel.basic_ack(mtd.delivery_tag)
            pa.append(gradv)
            newkl = -torch.mul(gradv, v).sum()
            newbl = {}
            for nd in NODES:
                newbl[nd] = 2 * (ws[nd] - h2y[nd])
                newkl = newkl + torch.norm(ws[nd] - h2y[nd], 2).pow(2) - torch.mul(newbl[nd], ws[nd]).sum()
            pb.append(newbl)
            pc.append(2 * (z - h2z))
            newkl = newkl + torch.norm(z - h2z, 2).pow(2) - torch.mul(pc[-1], z).sum()
            pk.append(newkl)
            lbda.append(float(os.environ['lbda_init']))

            # update cp sync
            for nd in NODES:
                b_nd = torch.stack([k[nd] for k in pb])
                channel.basic_publish('', nd, body=pack_tensors('main', b=b_nd), properties=propWithModeHeader('update_b'))
            # continue step outer
            lbda_nd = torch.tensor(lbda) if len(lbda) > 0 else None
            for nd in NODES:
                channel.basic_publish('', nd, body=pack_tensors('main', lbda=lbda_nd, theta=thetas[nd]), properties=propWithModeHeader('step_outer'))
                taus[nd] = 1
        
        if (k+1) % pre_epochs == 0:
            # test loss acc
            test_loss, test_acc  = test_avg_loss(t_imgs, t_lbls, z)
            elapsed = time.time() - starttime
            starttime = time.time()
            Favg = sum(fi.values()) / len(fi)
            logger.info(f'eph = {k+1:5d}, test_loss = {test_loss:6.4f}, test_acc = {100*test_acc:7.4f}, Favg = {Favg:6.4f}, elapsed = {elapsed:7.1f}')
            eph = loss_history[-1][3] + (pre_epochs + phix_iter)
            res = [test_loss.item(), test_acc.item(), loss_history[-1][2] + elapsed, eph]
            train_log.loc[len(train_log)] = res
            LOG_INTERVAL = int(os.environ['log_interval'])
            if (k+1) % LOG_INTERVAL == 0:
                save(exp_time, k+1, train_log)
            loss_history.append(res)


    loss_history = np.array(loss_history)
    str = io.StringIO()
    np.savetxt(str, loss_history, fmt='%9.7f  %6.4f  %f  %d')
    print(str.getvalue())
    print(f'v_lr = {v_lr}, z_lr = {z_lr}, lbda_lr = {lbda_lr}, theta_lr = {theta_lr}, phi_lr = {phi_lr}')
    print(f'pre_epochs = {pre_epochs}, S = {S}, tau = {tau}, active_set = {active_set}, phix_iter = {phix_iter}')
    print(f"c1 = {float(os.environ['c1'])}, c2 = {float(os.environ['c2'])}")

    # we're done
    channel.cancel()


def crash_track(main):
    try:
        main()
    except Exception:
        logging.exception("main-error")


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout)
    crash_track(main)
