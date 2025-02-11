# Blackout DIFUSCO: Advancing Diffusion-Based Optimization for Combinatorial Problems

This paper has been archived at [arXiv:2502.05221](https://arxiv.org/abs/2502.05221).

![Blackout Diffusion](forward_blackout.gif)

## Abstract
This study investigates enhancements to the DIFUSCO model, a diffusion-based framework for solving combinatorial optimization problems. We propose a series of novel contributions aimed at improving performance and computational efficiency: integrating Blackout Diffusion for continuous-time dynamics, implementing a linear noise scheduling strategy for stability, and introducing heuristic sampling to reduce computational overhead while maintaining solution quality. Our experimental results demonstrate the effectiveness of these methods and provide new insights into balancing simplicity and complexity in model design.

---

## Introduction
Diffusion-based models have shown significant promise in solving combinatorial optimization problems, such as the Traveling Salesman Problem (TSP). DIFUSCO, a foundational model in this domain, demonstrates strong baseline performance. However, its scalability and robustness in handling complex dynamics warrant further exploration.

Our contributions include:
1. Leveraging **Blackout Diffusion** to better model continuous-time dynamics.
2. Introducing **linear noise scheduling** for more systematic noise control and enhanced model stability.
3. Applying **heuristic sampling techniques** to optimize computational efficiency without sacrificing performance.

This paper evaluates these contributions through extensive experimentation and analysis, highlighting both the strengths and limitations of the proposed approaches.

---

## Methodology
### Model Enhancements
1. **Continuous-Time Dynamics with Blackout Diffusion**  
   - We incorporated Blackout Diffusion, which improves the model’s ability to handle continuous dynamics by mitigating discretization errors. This enhancement is particularly useful for highly dynamic optimization problems.
   
2. **Noise Stabilization via Linear Scheduling**  
   - To address instability during training, we applied a linear noise scheduling mechanism. This method ensures a gradual and systematic introduction of noise, leading to smoother convergence.

3. **Computational Optimization with Heuristic Sampling**  
   - We developed a heuristic sampling strategy that selectively focuses on the 50 most promising points during optimization. This approach reduces computational costs significantly while maintaining solution quality.

---

## Experimental Setup
### Dataset
We used the following TSP datasets for training and evaluation:
- **Training Set**: TSP-50 and TSP-100 (1,502,000 instances), TSP-500 (128,000 instances).
- **Test Set**: TSP-50 and TSP-100 (1,280 instances), TSP-500 (128 instances).

### Training Protocol
Each model was trained for 50 epochs with a batch size of 64 (TSP-50), 16 (TSP-100), and 4 (TSP-500). Loss and solution cost (tour distance) were measured to evaluate performance.

---

## Results and Discussion
### Key Findings
- **Performance Trends**: 
  - Incorporating Blackout Diffusion, noise scheduling, and heuristic sampling improved performance incrementally, demonstrating the effectiveness of each enhancement.
  - Despite these improvements, the original DIFUSCO model outperformed all modified versions in overall robustness and simplicity.

- **Trade-Off Analysis**:  
  - While enhancements added complexity, they introduced diminishing returns. The simplicity of DIFUSCO allowed it to remain competitive in both performance and computational requirements.

### Performance Metrics
| Model Variant      | Epoch (Best) | Validation Loss | Solved Cost (Tour Distance) |
|--------------------|--------------|-----------------|-----------------------------|
| Blackout Diffusion | 46           | 5.7447          | 5.7452                      |
| + Noise Scheduling | 36           | 5.7447          | 5.745                      |
| + Heuristic Sampling | 45         | 5.7449          | 5.7454                      |
| DIFUSCO (Baseline) | 47           | 5.7445          | **5.744**                   |

---

## Implications and Contributions
This work provides the following contributions:
1. A detailed investigation into enhancing diffusion models for combinatorial optimization through continuous dynamics and noise management.
2. A novel heuristic sampling technique that significantly reduces computational costs, paving the way for more scalable optimization frameworks.
3. Insights into the trade-off between model complexity and performance, emphasizing the importance of simplicity in robust model design.

---

## Future Work
1. **Objective Function Guidance**: Incorporate domain-specific heuristics to further refine optimization objectives.
2. **Scalability Improvements**: Test on larger datasets with increased state space.
3. **Hardware Utilization**: Address current computational bottlenecks by leveraging additional GPUs to expedite experimentation.

---

## Conclusion
While our proposed enhancements showed incremental improvements, the original DIFUSCO model demonstrated unmatched robustness and simplicity, highlighting the challenges of balancing model complexity with practical performance. This study lays the groundwork for future research in diffusion-based combinatorial optimization.

---

## Acknowledgments
This research was made possible through continuous access to NVIDIA RTX 3090 GPUs and the collaborative efforts of Team 3: Jun Pyo Seo, Han Joon Byun, Seonjun Kim, Euijun Jung, and Nadine Ben Amar.

