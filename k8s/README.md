# Deploying to Minikube (Task 7)

```bash
# 1. Start the cluster
minikube start

# 2. Build the image *inside* Minikube's Docker daemon so the cluster can
#    find it without needing a registry push
eval $(minikube docker-env)       # PowerShell: & minikube -p minikube docker-env --shell powershell | Invoke-Expression
docker build -f docker/Dockerfile -t cardiorisk-api:local .

# 3. Apply the manifests
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# 4. Check rollout status
kubectl get pods -l app=cardiorisk-api
kubectl get svc cardiorisk-api-service

# 5. Expose the LoadBalancer service locally (Minikube has no real cloud LB)
minikube service cardiorisk-api-service --url
# -> use the printed URL, e.g. curl http://127.0.0.1:PORT/health

# 6. Smoke-test the endpoint
curl -X POST http://127.0.0.1:PORT/predict \
  -H "Content-Type: application/json" \
  -d '{"age":57,"sex":1,"cp":4,"trestbps":140,"chol":289,"fbs":0,"restecg":2,"thalach":145,"exang":1,"oldpeak":2.0,"slope":2,"ca":1,"thal":7}'
```

Take screenshots of steps 4-6 for the report (`screenshots/` folder) - rollout status, service URL, and the curl response.

To tear down: `kubectl delete -f k8s/service.yaml -f k8s/deployment.yaml`.
