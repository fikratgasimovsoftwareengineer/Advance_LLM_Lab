# FSDP2 Kubernetes Multi-Node Lab

Obiettivo: training distribuito reale su due nodi Kubernetes, una GPU per nodo.

- 2 x EC2 g4dn.xlarge
- 1 x NVIDIA T4 per nodo
- k3s
- NVIDIA Device Plugin
- un'unica immagine Docker condivisa via ECR
- 2 Kubernetes Job: rank 0 e rank 1
- Qwen2.5-0.5B-Instruct
- subset Alpaca
- 1 epoca iniziale; massimo 5

## Ordine di lavoro

1. Build locale dell'immagine.
2. Test locale solo import Python/PyTorch.
3. Push immagine su ECR.
4. Creazione delle due EC2.
5. Installazione driver NVIDIA + NVIDIA Container Toolkit.
6. Creazione cluster k3s a due nodi.
7. Installazione NVIDIA Device Plugin.
8. Verifica `nvidia.com/gpu: 1` su entrambi i nodi.
9. Label nodi:
   `kubectl label node <NODE1> fsdp-rank=0`
   `kubectl label node <NODE2> fsdp-rank=1`
10. Sostituzione `REPLACE_ECR_URI` nei manifest.
11. Avvio service rendezvous + due Job.
12. Test con 1 epoca.
13. Solo dopo, passaggio a massimo 5 epoche.

## Build locale

```bash
docker build -t fsdp2-llm-lab:latest .
docker run --rm fsdp2-llm-lab:latest
```

## Training Kubernetes

```bash
kubectl apply -f k8s/00-master-service.yaml
kubectl apply -f k8s/10-rank0-job.yaml
kubectl apply -f k8s/20-rank1-job.yaml
```

Monitoraggio:

```bash
kubectl get pods -o wide -w
kubectl logs -f job/fsdp-rank0
kubectl logs -f job/fsdp-rank1
```
