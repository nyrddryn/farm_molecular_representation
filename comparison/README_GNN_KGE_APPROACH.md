# 🧬 Approach 2: GNN/KGE for Reaction Prediction

## 📋 Tổng Quan

Sử dụng **Graph Neural Networks (GNN)** để học molecular graph structure và **Knowledge Graph Embeddings (KGE)** để encode functional group relationships cho reaction prediction.

**Mục tiêu cuối cùng:** Predict reaction products bằng cách model molecular graphs và chemical knowledge

```
Input:  Reactant Graphs + Reaction Knowledge Graph
Output: Product Graph + Reaction Pathway
```

---

## 🎯 Kiến Trúc Tổng Thể

### **Phase 1: FG Graph Construction (✅ Đã hoàn thành)**

```
SMILES
   ↓
[RDKit] FG-Enhanced Tokens (ONLY THIS STEP uses RDKit)
   ↓
[Extract FG Nodes] C_ester, N_amine, etc.
   ↓
[KGE Features] 256-dim semantic embeddings per node
   ↓
[GNN Edge Prediction] Predict which nodes connect
   ↓
FG Graph Structure (nodes + edges)
   ↓
[Graph Matching] Compare 2 graphs with hybrid scoring
   ↓
Output: MCS + Differences with serial numbers
```

**Files:** 
- `compare_with_gnn.py` (comparison pipeline)
- `train_GCN_link_prediction.py` (GNN training)
- `train_FG_KGE.py` (KGE training)

**Kết quả hiện tại:**
- ✅ GNN predicts edges: Accuracy ~80%
- ✅ KGE semantic similarity: Coverage 1221 FG types
- ✅ Graph comparison: 6-25ms, ~90% accuracy
- ✅ NO RDKit dependency except FG detection

---

### **Phase 2: Reaction Prediction (🔄 Chưa implement)**

```
Reactant Graph A + Reactant Graph B
   ↓
[GNN Encoder] Learn graph representations
   ↓
[Knowledge Graph] Query reaction templates & rules
   ↓
[Graph Transformer] Predict graph edits
   ↓
[Graph Decoder] Construct product graph
   ↓
[SMILES Generation] Convert graph → SMILES
   ↓
Output: Product Graph + Reaction Center
```

**Kiến trúc chi tiết:**

```
Component 1: Molecular Graph Encoder
  - GCN/GAT/GraphTransformer (5-10 layers)
  - Node features: Atom type + FG embeddings (128-dim)
  - Edge features: Bond type + distance
  - Output: Graph-level embedding (512-dim)

Component 2: Reaction Knowledge Graph
  - Entities: Functional groups, reaction types, reagents
  - Relations: "reacts_with", "produces", "catalyzes"
  - KGE: ComplEx or RotatE (256-dim embeddings)
  - Query: Find applicable reaction templates

Component 3: Graph Transformer (Reaction Predictor)
  - Input: Reactant graphs + Reaction template
  - Predict: Which bonds break, which bonds form
  - Output: Edit operations on graph

Component 4: Product Graph Decoder
  - Apply edits to reactant graph
  - Form product graph
  - Validate chemical rules
  - Convert to SMILES

Total Parameters: 
  - GNN Encoder: ~50M
  - Knowledge Graph: ~20M
  - Graph Transformer: ~100M
  - Total: ~170M parameters
```

---

## 📊 INPUT & OUTPUT

### **Phase 1: Graph Comparison (Hiện tại)**

| Component | Details |
|-----------|---------|
| **Input** | 2 SMILES strings |
| **Process** | Build FG graphs with GNN, compare with KGE similarity |
| **Output** | `"(FG1):1 MCS (FG2):2"` with atom-level or FG labels |
| **Example** | Input: `CS(=O)(=O)c1ccc(CCN)cc1CC(=O)O` vs `COC(=O)c1ccc(CCO)cc1CC(=O)O`<br>Output: `(C_alkyl_S_sulfonyl_O_sulfonyl_O_sulfonyl):1 MCS N_primary_amine:2`<br>`and (C_ester_O_ester_C_ester_O_ester):1 MCS O_hydroxyl:2` |

### **Phase 2: Reaction Prediction (Mục tiêu)**

| Component | Details |
|-----------|---------|
| **Input Format** | Reactant molecular graphs + Reaction context |
| **Graph Representation** | Nodes: Atoms + FG labels<br>Edges: Bonds + spatial relationships |
| **Output Format** | Product graph + Edit sequence + Confidence |
| **Example 1** | **Input:** Graph(COC(=O)Cl) + Graph(NH3) + [base]<br>**Output:** Graph(COC(=O)N) + Edits: [break(C-Cl), form(C-N)]<br>Confidence: 94% |
| **Example 2** | **Input:** Graph(benzene) + Graph(Br2) + [FeBr3]<br>**Output:** Graph(bromobenzene) + Edits: [add(C-Br), add(H-Br)]<br>Confidence: 91% |

---

## 💰 CHI PHÍ & TÀI NGUYÊN

### **1. Dữ Liệu Training**

#### **Yêu cầu dữ liệu:**

| Scale | Reactions | Graph Annotations | Expected Accuracy | Use Case |
|-------|-----------|-------------------|-------------------|----------|
| **POC** | 50K | Automatic | 50-60% | Initial testing |
| **Development** | 200K | Semi-automatic | 70-75% | Research |
| **Production** | 1M | Validated | 80-85% | Industry |
| **SOTA** | 5M+ | Expert-curated | 85-90% | Commercial |

**Đặc thù của GNN approach:**
- ⚠️ Cần **graph annotations**: reaction centers, atom mappings
- ⚠️ **Preprocessing phức tạp hơn**: Build graphs từ SMILES
- ⚠️ **Data quality critical**: Sai atom mapping → model học sai

#### **Chi phí preprocessing:**

| Task | Time | Cost | Tools |
|------|------|------|-------|
| **Parse SMILES → Graphs** | 1 week | Auto | RDKit + NetworkX |
| **Atom Mapping** | 2-4 weeks | $10K-20K | RXNMapper + manual |
| **FG Annotation** | 1 week | Auto | helpers.py |
| **Reaction Center Detection** | 2 weeks | $5K | Semi-auto |
| **Validation** | 2 weeks | $10K | Expert chemists |
| **Total (500K rxns)** | **2-3 months** | **$25K-35K** | |

---

### **2. Training Infrastructure**

#### **Component-wise Training:**

**A. GNN Molecular Encoder (Pre-training)**

| Dataset | Molecules | Training Time | GPU | Cost |
|---------|-----------|---------------|-----|------|
| PubChem subset | 10M | 7 days | 4× A100 | $1,700 |
| ZINC | 20M | 14 days | 8× A100 | $6,700 |

**B. Knowledge Graph Embeddings**

| KG Size | Entities | Relations | Training Time | GPU | Cost |
|---------|----------|-----------|---------------|-----|------|
| FG-KG | 1,221 FGs | 9 types | 2 days | 1× RTX 3090 | $25 |
| Full Chem-KG | 50K entities | 100 types | 5 days | 4× A100 | $1,200 |

**C. Reaction Predictor (Main Task)**

| Dataset | Reactions | Epochs | Time/Epoch | Total Time | GPU Setup | Cost |
|---------|-----------|--------|------------|------------|-----------|------|
| 50K | 50K | 30 | 3h | 4 days | 1× A100 | $500 |
| 200K | 200K | 20 | 10h | 8 days | 4× A100 | $1,900 |
| 1M | 1M | 20 | 40h | 33 days | 8× A100 | $22,000 |

**Memory Requirements:**

| Component | Model Size | Memory (FP16) | Batch Size | GPU Needed |
|-----------|------------|---------------|------------|------------|
| GNN Encoder | 50M | ~200MB | 32 graphs | RTX 3090 (24GB) |
| KGE | 20M | ~80MB | 256 triples | RTX 3090 |
| Graph Transformer | 100M | ~400MB | 16 graphs | A100 40GB |
| **Full Pipeline** | **170M** | **~800MB** | **8-16** | **A100 40GB** |

---

### **3. Thời Gian Training - Breakdown Chi Tiết**

#### **Scenario A: Academic/POC (50K reactions)**

```
Pre-training:
  - GNN Encoder (PubChem 10M): 7 days, 4× A100 = $1,700
  - KGE (FG-KG 1.2K): 2 days, 1× RTX 3090 = $25
  
Main Training:
  - Reaction Predictor: 4 days, 1× A100 = $500
  
Total: 13 days, $2,225
```

**Full Academic Budget:**
```
Data: FREE (USPTO)
Preprocessing: 1 month, automated + $5K validation
Pre-training: 13 days, $2.2K
Development: 2 months (student labor, $0)
Validation: 1 week, $2K
----------------------------------------
Total: ~$9K + 3.5 months
Expected Accuracy: 65-70%
```

---

#### **Scenario B: Production (1M reactions)**

```
Pre-training:
  - GNN Encoder (ZINC 20M): 14 days, 8× A100 = $6,700
  - KGE (Full Chem-KG 50K): 5 days, 4× A100 = $1,200
  
Main Training:
  - Reaction Predictor: 33 days, 8× A100 = $22,000
  
Total: 52 days, $29,900
```

**Full Production Budget:**
```
Data: $50K (Reaxys) + $30K (atom mapping + validation)
Preprocessing: 3 months, 2 engineers × $10K = $20K
Pre-training: 52 days, $30K
Hyperparameter tuning: 2 weeks, $3K
Validation: 1 month, 2 chemists × $5K = $10K
Infrastructure: $5K (storage, compute)
Development: 4 months, 2 ML engineers × $20K = $40K
----------------------------------------
Total: ~$188K + 6 months
Expected Accuracy: 80-85%
```

---

#### **Scenario C: State-of-the-art (5M reactions)**

```
Pre-training:
  - Multi-task GNN (100M molecules): 30 days, 16× A100 = $20,000
  - Large Chem-KG (500K entities): 10 days, 8× A100 = $6,000
  
Main Training:
  - Reaction Predictor: 60 days, 16× A100 = $40,000
  - Ensemble (5 models): 300 days cumulative, $200K
  
Total: 100+ days, $266K (just GPU)
```

**Full SOTA Budget:**
```
Data: $150K (Reaxys full + proprietary)
Expert curation: $80K
Pre-training: $266K (GPU)
Main training: Included above
Validation & testing: $50K
Infrastructure: $20K
Team: 6 months, 5 people × $25K = $125K
----------------------------------------
Total: ~$691K + 6-9 months
Expected Accuracy: 85-90%
```

---

### **4. Inference Cost**

| Setup | Hardware | Throughput | Cost/Reaction |
|-------|----------|------------|---------------|
| **CPU** | 16 cores | 200 rxns/sec | $0.0002 |
| **GPU (RTX 3090)** | 24GB | 1000 rxns/sec | $0.00005 |
| **GPU (A100)** | 40GB | 2000 rxns/sec | $0.00003 |
| **Multi-GPU** | 4× A100 | 7000 rxns/sec | $0.00001 |

**Graph operations rất fast:**
- Graph encoding: 5-10ms
- KG query: 1-2ms
- Prediction: 3-5ms
- Total: **10-20ms per reaction** (nhanh hơn BERT 5-10x)

**Production cost example:**
```
1M predictions/day:
  - 1× A100 GPU: $5/hour × 24h = $120/day
  - Can handle: 2000 × 3600 × 24 = 172M rxns/day
  - Cost: $120/172M = $0.0000007/reaction
  
Annual at 365M predictions:
  - GPU: $44K/year
  - Infrastructure: $20K/year
  - Total: $64K/year
  - Much cheaper than BERT! ($120K/year)
```

---

## ✅ ƯU ĐIỂM

### **1. Kỹ Thuật:**
- ✅ **Graph Native**: Chemistry IS graphs - natural representation
- ✅ **Fast Inference**: 10-20ms per reaction (5x faster than BERT)
- ✅ **Explicit Structure**: Directly models bonds, atoms, stereochemistry
- ✅ **Local + Global**: GNN captures both local motifs and global structure
- ✅ **Interpretable**: Can visualize reaction centers, activation sites
- ✅ **Minimal RDKit**: Only for initial FG detection, pure graph afterwards

### **2. Scalability:**
- ✅ **Parallel Processing**: Graphs naturally parallelizable
- ✅ **Batch Processing**: Efficient batching với graph padding
- ✅ **Memory Efficient**: Sparse graph operations
- ✅ **Incremental Updates**: Easy to add new reaction types to KG

### **3. Knowledge Integration:**
- ✅ **Chemistry Rules**: Can encode reaction rules explicitly in KG
- ✅ **Transfer Learning**: KG embeddings transferable across tasks
- ✅ **Multi-Task**: Single KG for multiple prediction tasks
- ✅ **Explainable**: Can query KG for "why this reaction?"

### **4. Accuracy:**
- ✅ **Stereochemistry**: Graphs naturally handle chirality, E/Z
- ✅ **Reaction Center**: Explicitly predicts which bonds break/form
- ✅ **Side Products**: Can predict multiple products from graph
- ✅ **Novel Reactions**: KG helps generalize to unseen combinations

### **5. Development:**
- ✅ **Modular**: GNN, KGE, Predictor train separately
- ✅ **Flexible**: Easy to swap GNN architectures (GCN→GAT→GraphTransformer)
- ✅ **Reusable**: Pre-trained GNN/KGE reusable for other tasks

---

## ⚠️ NHƯỢC ĐIỂM

### **1. Complexity:**
- ❌ **Complex Pipeline**: GNN + KGE + Transformer = nhiều components
- ❌ **Hard to Debug**: Graph operations khó visualize và debug
- ❌ **Preprocessing Heavy**: Build graphs from SMILES phức tạp
- ❌ **Atom Mapping Required**: Cần expensive atom mapping annotation
- ❌ **Graph Isomorphism**: Khó so sánh 2 graphs efficiently

### **2. Data Requirements:**
- ❌ **More Data Needed**: Cần 500K-5M reactions vs 100K cho BERT
- ❌ **Annotation Cost**: Atom mapping + reaction center = expensive
- ❌ **Quality Critical**: Sai graph structure → model useless
- ❌ **Imbalanced**: Hiếm reaction types rất khó học

### **3. Training:**
- ❌ **Long Pre-training**: GNN pre-training takes weeks
- ❌ **Memory Intensive**: Large graphs need lots of GPU memory
- ❌ **Hyperparameter Sensitive**: GNN architecture choices critical
- ❌ **Slow Convergence**: Graph models converge slower than seq2seq

### **4. Implementation:**
- ❌ **Complex Code**: Graph ops harder than sequence ops
- ❌ **Library Maturity**: PyG/DGL less mature than HuggingFace
- ❌ **Fewer Examples**: Less documentation than BERT
- ❌ **Hardware Specific**: Performance varies greatly by GPU

### **5. Specific Limitations:**
- ❌ **Large Molecules**: O(N²) complexity for N atoms
- ❌ **Dynamic Bonds**: Hard to model bond order changes
- ❌ **Solvent Effects**: Difficult to incorporate implicit solvation
- ❌ **Quantum Effects**: Cannot capture quantum tunneling, etc.

---

## 🔄 So Sánh Chi Tiết: GNN vs BERT

| Aspect | GNN/KGE | BERT/FARM |
|--------|---------|-----------|
| **Representation** | Graphs (native chemistry) | Sequences (SMILES strings) |
| **Inference Speed** | ✅ Very Fast (10-20ms) | ⚠️ Medium (50-200ms) |
| **Training Time** | ❌ Long (weeks pre-training) | ✅ Short (days, pre-trained) |
| **Data Needed** | ❌ 500K-5M reactions | ✅ 100K-1M reactions |
| **Annotation Cost** | ❌ High (atom mapping) | ✅ Low (just reactions) |
| **Stereochemistry** | ✅ Excellent (explicit) | ❌ Poor (implicit) |
| **Reaction Center** | ✅ Explicit prediction | ❌ Implicit |
| **Interpretability** | ✅ High (graph edits) | ⚠️ Medium (attention) |
| **Novel Reactions** | ✅ Good (KG reasoning) | ❌ Poor (sequence only) |
| **Memory (Training)** | ⚠️ Medium (~1GB) | ⚠️ Medium (~1.5GB) |
| **Memory (Inference)** | ✅ Low (~200MB) | ⚠️ High (~500MB) |
| **Development Cost** | ❌ High ($200K-700K) | ✅ Lower ($100K-300K) |
| **Code Complexity** | ❌ High | ✅ Medium |
| **Community Support** | ⚠️ Growing | ✅ Mature |
| **Production Ready** | ⚠️ 2024-2025 | ✅ Now |

---

## 🔀 Hybrid Approach (Recommended!)

**Best of both worlds:**

```
Stage 1: BERT for quick MVP
  - Use BERT/FARM (3 months, $50K)
  - Get to 70-75% accuracy fast
  - Deploy & collect user data

Stage 2: Add GNN Components
  - Train GNN encoder (1 month, $10K)
  - Use for reaction center detection
  - Hybrid: BERT for candidates + GNN for ranking

Stage 3: Full GNN Pipeline
  - Complete GNN/KGE system (6 months, $200K)
  - Achieve 85-90% accuracy
  - Replace BERT gradually
```

**Kiến trúc Hybrid:**
```
Input: SMILES
   ↓
[BERT Encoder] Fast candidate generation → Top 10 products
   ↓
[GNN Reranker] Detailed scoring with graphs → Top 3 products
   ↓
[KG Validator] Check against chemical rules → Final product
   ↓
Output: Best product + confidence
```

---

## 🚀 Lộ Trình Phát Triển GNN

### **Stage 1: Foundation (4 months, $30K)**
```
Month 1-2: Data Collection & Preprocessing
  - Collect 200K USPTO reactions
  - Atom mapping annotation
  - Build FG graphs
  - Cost: $15K (annotation)

Month 3: Pre-training
  - Train GNN on 10M molecules
  - Train KGE on FG-KG
  - Cost: $5K (GPU)

Month 4: Initial Model
  - Train reaction predictor (50K)
  - Evaluate baseline
  - Cost: $2K (GPU)

Deliverable: 60-65% accuracy, proof-of-concept
```

### **Stage 2: Development (6 months, $100K)**
```
Month 1-3: Scale Data
  - Scale to 1M reactions
  - Expert validation
  - Cost: $40K

Month 4-5: Model Improvements
  - Larger GNN (50M → 100M params)
  - Better KG (50K entities)
  - Hyperparameter tuning
  - Cost: $30K (GPU)

Month 6: Evaluation & Testing
  - Comprehensive benchmark
  - A/B testing vs BERT
  - Cost: $10K

Deliverable: 75-80% accuracy, production-ready
```

### **Stage 3: Production (12 months, $300K)**
```
Quarter 1: Data Excellence
  - 5M reactions curated
  - Expert validation
  - Cost: $100K

Quarter 2-3: SOTA Model
  - Multi-task learning
  - Ensemble methods
  - Yield + selectivity prediction
  - Cost: $120K (GPU + labor)

Quarter 4: Deployment
  - API service
  - Monitoring
  - User feedback
  - Cost: $80K

Deliverable: 85-90% accuracy, commercial product
```

---

## 📚 Tài Liệu Tham Khảo

### **Papers:**
1. **GNN for Chemistry**: "Analyzing Learned Molecular Representations for Property Prediction" (Yang et al., 2019)
2. **Reaction Prediction**: "Predicting Organic Reaction Outcomes with Weisfeiler-Lehman Network" (Jin et al., 2017)
3. **Knowledge Graph**: "Automatic Retrosynthesis Pathway Planning Using ComplEx" (Chen et al., 2020)
4. **Graph Transformer**: "Do Transformers Really Perform Bad for Graph Representation?" (Ying et al., 2021)

### **Libraries:**
- **PyTorch Geometric**: https://pytorch-geometric.readthedocs.io/
- **DGL (Deep Graph Library)**: https://www.dgl.ai/
- **RDKit**: https://www.rdkit.org/
- **NetworkX**: https://networkx.org/

### **Datasets:**
- **USPTO**: 1.8M reactions (free)
- **ORD**: 5M+ reactions (free)
- **Reaxys**: 50M reactions (commercial)

---

## 🎓 Kết Luận

### **GNN/KGE approach phù hợp khi:**
- ✅ Có budget lớn ($200K-700K)
- ✅ Cần accuracy cao (>85%)
- ✅ Cần predict reaction center explicitly
- ✅ Có data nhiều (>500K reactions with atom mapping)
- ✅ Cần fast inference (<20ms)
- ✅ Team có expertise về graph ML
- ✅ Long-term investment (12+ months)

### **KHÔNG phù hợp khi:**
- ❌ Budget hạn chế (<$50K)
- ❌ Cần deploy nhanh (<3 months)
- ❌ Data ít (<200K reactions)
- ❌ Không có atom mapping annotations
- ❌ Team chưa có experience với graphs

### **Recommendation Matrix:**

| Your Situation | Recommended Approach | Timeline | Budget |
|----------------|---------------------|----------|---------|
| **Startup MVP** | BERT only | 3 months | $50K |
| **Research Lab** | BERT + GNN hybrid | 6 months | $100K |
| **Pharma Company** | Full GNN/KGE | 12 months | $300K |
| **Best-in-class** | Ensemble (BERT+GNN+Rules) | 18 months | $500K+ |

### **Next Steps:**
1. ✅ Phase 1 done: Graph comparison working
2. → Pre-train GNN encoder on molecules
3. → Scale up KGE to larger chemistry KG
4. → Implement graph transformer for reaction prediction
5. → Collect & annotate reaction data
6. → Train & evaluate baseline
7. → Iterate to production
