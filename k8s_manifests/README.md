## Kubernetes deployment (HW4)
### (working with online inference app (done in HW2))
Install **kubectl**:
```
brew install kubectl
```
Install **minikube**:
 ```
 brew install minikube
```
Start cluster: 
```
minikube start
```
Enable port forwarding: 
```
kubectl port-forward pod/online-inference 8000:8000
```
Apply any manifest:
```
kubectl apply -f <MANIFEST_FILENAME>.yaml
```