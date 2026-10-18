# Deploying to Minikube (Task 7)

```bash
# 1. Start the cluster
minikube start --driver=docker

# 2. Build the image locally and copy it into the cluster, so no registry
#    push is needed (recent Minikube uses containerd, so `minikube
#    docker-env` no longer works for this)
docker build -f docker/Dockerfile -t cardiorisk-api:local .
minikube image load cardiorisk-api:local

# 3. Apply the manifests
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# 4. Check rollout status
kubectl get pods -l app=cardiorisk-api
kubectl get svc cardiorisk-api-service

# 5. Expose the LoadBalancer service locally (Minikube has no real cloud LB)
minikube service cardiorisk-api-service --url   # on Windows, keep this terminal open (it's the tunnel)
# -> use the printed URL, e.g. curl http://127.0.0.1:PORT/health

# 6. Smoke-test the endpoint
curl -X POST http://127.0.0.1:PORT/predict \
  -H "Content-Type: application/json" \
  -d '{"age":57,"sex":1,"cp":4,"trestbps":140,"chol":289,"fbs":0,"restecg":2,"thalach":145,"exang":1,"oldpeak":2.0,"slope":2,"ca":1,"thal":7}'
```

Take screenshots of steps 4-6 for the report (`screenshots/` folder) - rollout status, service URL, and the curl response.

To tear down: `kubectl delete -f k8s/service.yaml -f k8s/deployment.yaml`.
