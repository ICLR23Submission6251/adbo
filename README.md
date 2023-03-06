# Asynchronous Distributed Bilevel Optimization

Codes for ICLR 2023 paper [Asynchronous Distributed Bilevel Optimization](https://arxiv.org/pdf/2212.10048.pdf)

## Run

We share our experiment code for data hyper-cleaning on MNIST and regularization coefficient optimization on IJCNN1. 

In the parameter server architecture we utilized, our master and worker nodes run as docker containers and communicate through RabbitMQ. 

To run locally, build the docker images for master and worker nodes and invoke docker compose to start the containers accordingly:

```bash
docker build cp-node-async -t cp-node-async
docker build cp-main-async -t cp-main-async
docker compose up
```

To change experiment settings, edit `cp-exp.env` with your wanted parameters. We explain their meanings as follows:

- `w_lr`: local lower-level variable learning rate on workers
- `hp_lr`: local upper-level variable learning rate on workers
- `z_lr`: consensus lower-level variable learning rate on master
- `v_lr`: consensus upper-level variable learning rate on master
- `lbda_init`: initial dual variable value for cutting-plane constraints
- `pre_epochs`: number of iterations between cutting plane updates
- `S`: minimum number of active workers required to perform an update
- `tau`: maximum number of iterations between worker communications with the master
- `active_set`: whether to automatically remove inactive cutting planes

## Citation

```tex
@inproceedings{
    jiao2023asynchronous,
    title={Asynchronous Distributed Bilevel Optimization},
    author={Yang Jiao and Kai Yang and Tiancheng Wu and Dongjin Song and Chengtao Jian},
    booktitle={The Eleventh International Conference on Learning Representations },
    year={2023},
    url={https://openreview.net/forum?id=_i0-12XqVJZ}
}
```
