1. Install **kubectl**:
`brew install kubectl`
2. Install **minikube**: `brew install minikube`
3. Start cluster: `minikube start`
4. Enable port forwarding: `kubectl port-forward pod/online-inference 8000:8000
`