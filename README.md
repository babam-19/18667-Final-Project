# Cluster-Comm: A Semidecentralized Federated Learning Network

> **A scalable, cluster-based hybrid communication framework to improve convergence stability and reduce communication overhead in distributed learning.**

<div align="center">
  
| **Jana Armouti** | **Brianna Abam** |
|------------------|------------------|
| Carnegie Mellon University | Carnegie Mellon University |
| jarmouti@andrew.cmu.edu | babam@andrew.cmu.edu |


<img width="740" height="598" alt="cluster-comm-architecture" src="https://github.com/user-attachments/assets/1e814d53-9ebe-47a1-8f59-28ddff8b17c0" />

</div>

## **1. Abstract/Overview**
Efficient communication is a key challenge in gossip networks used for synchronous
SGD, primarily due to scalability bottlenecks as the number of clients grows. While
decentralized SGD can mitigate communication overhead by removing the central
parameter server, it often suffers from slower convergence and instability in large,
sparse systems. To address these limitations, we propose **Cluster-Comm**, a 
semi-decentralized framework that groups clients into clusters for hybrid communication 
with frequent local updates within clusters and periodic synchronization with a parameter server.
This approach aims to balance scalability and convergence stability, reducing com-
munication overhead while maintaining efficient training across large distributed
or edge-based learning systems.

---

## **2. Method Overview**

Our semidecentralized architecture is composed of:

### **2.1 Cluster-based Gossiping/Local Synchronization**
Clients are grouped into clusters and within each cluster, clients exchange model updates 
after a certain number of iterations, with hopes of amortizing communication cost.

### **2.2 Periodic Global Synchronization**
Cluster representatives periodically communicate with a central 
parameter server. Under various test constraints, we hope to see this global synchronization:

- improve convergence stability,
- prevent significant model drift, and  
- maintain alignment across clusters

### **2.3 Experimental Evaluation**
Our codebase includes:

- Simulation of our semidecentralized FL networks (along with centralized fully synchronous SGD, centralized local-iteration
SGD, and fully decentralized gossip depending on the test)
- Code to reproduce similar results used to generate the experimental plots  
- Tools (through manipulation of the code) to explore the effects of cluster size, cluster count, graph topology, communication frequency, etc.

---

## **3. Cloning the repository**
```bash
git clone git@github.com:babam-19/18667-Final-Project.git
cd main
```

## **4. Running the tests**
```bash
python3 runner_file.py
```





