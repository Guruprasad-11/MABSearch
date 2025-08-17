# MAB Search: An Adaptive Learning Rate Optimization Framework

##  Academic Context

This project was developed as part of the **IIITDM Kancheepuram SIES Internship** under the mentorship of **Dr. A S Syed Shahul Hameed**. The work extends the original MABSearch algorithm, which was designed for benchmark function optimization, to machine learning model training scenarios.

###  Original Research
- **MABSearch Algorithm**: Originally developed by Dr. A S Syed Shahul Hameed and Narendran Rajagopalan
- **Paper**: [MABSearch: The Bandit Way of Learning the Learning Rate—A Harmony Between Reinforcement Learning and Gradient Descent](https://link.springer.com/article/10.1007/s40009-023-01292-1)
- **Code**: [GitHub Repository](https://github.com/Shahul-Rahman/MABSearch-Learning-the-learning-rate)

##  Project Overview

This project implements and compares a **Multi-Armed Bandit (MAB)** algorithm for adaptive learning rate selection in neural network training against traditional fixed learning rate strategies. The goal is to demonstrate how MAB can automatically select optimal learning rates during training to improve model performance and convergence.

### Key Contributions
- **Domain Extension**: Applied MABSearch to machine learning model training
- **Real-world Dataset**: Tested on California Housing Dataset instead of benchmark functions
- **Comparative Analysis**: Compared against fixed learning rates
- **Practical Implementation**: Demonstrated applicability in neural network training scenarios

## Dataset

**California Housing Dataset**
- **Source**: Scikit-learn built-in dataset
- **Features**: 8 numerical features (median income, house age, average rooms, etc.)
- **Target**: Median house value (in $100,000s)
- **Size**: ~20,640 samples
- **Type**: Regression problem

## Methodology

### Data Preprocessing
1. **Imputation**: Handle missing values using mean imputation
2. **Standardization**: Scale features to zero mean and unit variance
3. **Train-Test Split**: 80% training, 20% testing

### Model Architecture
- **Neural Network**: Single-layer perceptron (linear model)
- **Activation**: Identity function (linear regression)
- **Solver**: Stochastic Gradient Descent (SGD)
- **Loss Function**: Mean Squared Error (MSE)

### Learning Rate Strategies

#### Fixed Learning Rates
- Test multiple predefined learning rates: `[0.1, 0.01, 0.001, 0.0001]`
- Train each model independently
- Compare final performance

#### Multi-Armed Bandit (MAB) - Extended MABSearch
- **Arms**: Two learning rates `[0.1, 0.0001]`
- **Algorithm**: Epsilon-greedy with exponential decay
- **Exploration**: Random selection with probability ε
- **Exploitation**: Select best performing learning rate
- **Reward**: Training MSE (updated using exponential moving average)

## Implementation Details

### MAB Algorithm Parameters
```python
# Key Parameters
epsilon_decay = 0.1
max_epsilon = 1.0      # Initial exploration probability
min_epsilon = 0.001    # Final exploration probability
reward_alpha = 0.9     # Exponential moving average factor
steps = 100           # Total training steps
```

### Training Process
1. **Initialization**: Start with random learning rate
2. **Selection**: Choose learning rate using epsilon-greedy policy
3. **Training**: Update model weights for 1 iteration
4. **Evaluation**: Calculate MSE on training data
5. **Reward Update**: Update reward estimates for selected arm
6. **Epsilon Decay**: Reduce exploration probability
7. **Repeat**: Continue for 100 steps

## Results

### Performance Comparison
| Strategy | Test MSE | Performance |
|----------|----------|-------------|
| Fixed LR 0.1 | 0.563 | Baseline |
| Fixed LR 0.0001 | 5.891 | Poor |
| **MAB Algorithm** | **0.551** | **Best** |

### Visualizations

1. **Reward Value Tracking**: Shows how MAB estimates performance of each learning rate over time
2. **MSE Trend Comparison**: Compares training loss across different strategies
3. **Learning Rate Selection**: Demonstrates exploration vs exploitation balance

### Key Findings
- MAB achieves slightly better performance than the best fixed learning rate
- Algorithm tends to favor higher learning rates (0.1)
- Shows more stable convergence pattern
- Demonstrates effective exploration-exploitation balance

## Limitations and Scope

- Cannot surpass state-of-the-art adaptive optimizers like Adam and RMSprop.
- Higher computational overhead and requires careful hyperparameter tuning.
- May not scale efficiently to large datasets.
- Limited to simple optimization tasks, like learning rate selection.
- Not suitable for complex deep learning architectures.
- Performance depends heavily on the reward function design.

## Future Work

### Potential Improvements
1. **Advanced MAB Algorithms**: Implement UCB, Thompson Sampling
2. **Hyperparameter Optimization**: Extend to other hyperparameters
3. **Deep Learning**: Apply to more complex neural architectures
4. **Online Learning**: Real-time adaptation during deployment
5. **Multi-Objective**: Balance accuracy, training time, and resource usage

### Research Directions
- Compare with other adaptive learning rate methods (Adam, RMSprop)
- Investigate MAB for different types of neural networks
- Study the impact of reward function design
- Explore ensemble methods with MAB selection

## Legal and Copyright Notice

### Academic Use
This project is developed for academic research purposes as part of an internship program. The implementation extends the original MABSearch algorithm to new domains.

### Copyright Considerations
- **Original MABSearch**: Copyright belongs to Dr. A S Syed Shahul Hameed and Narendran Rajagopalan
- **This Extension**: Developed under academic supervision and NDA agreement
- **Dataset**: California Housing Dataset is publicly available through scikit-learn
- **Libraries**: All used libraries are open-source with appropriate licenses

### Usage Guidelines
- This code is provided for educational and research purposes
- Proper citation of the original MABSearch paper is required
- Commercial use may require additional permissions
- Users should respect the original authors' intellectual property rights

## License

This project is for academic research purposes. Please refer to the original MABSearch repository for licensing information regarding the core algorithm.

---

**Note**: This project extends the original MABSearch algorithm for educational and research purposes. Proper attribution to the original authors is essential when using or citing this work.
