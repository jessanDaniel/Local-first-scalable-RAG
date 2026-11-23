# Answer labels

**Labelled by Claude (Opus 5.5) on 2026-11-23, at the user's request (the user was busy). These are not
human labels; the user may review or overrule any of them.**

For each entry, fill in the lines at the bottom. Judge **only from the text shown**: the
reference passage is what the paper says; the quoted sources are what the model was given.

| Field | Values | Meaning |
|---|---|---|
| `verdict` | `correct` | answers the question in agreement with the reference passage |
| | `partial` | right in part: misses something asked for, or adds a wrong/unsupported detail |
| | `wrong` | the main answer disagrees with the reference passage |
| | `refused_ok` | marked "not in the corpus", and the model said it could not answer |
| | `refused_wrong` | the reference passage answers it, but the model said the sources do not |
| | `unsure` | the passage shown is not enough to judge (say why in `note`) |
| `[n] supports` | `yes` / `partly` / `no` | does source [n]'s quoted text say what the sentences citing it claim? |
| `note` | free text | anything odd: often the most useful line |

An answer can be `correct` while a citation `supports: no` (right fact, wrong source). Mark both.

<!-- entry: papers-20261123T062859Z-fused | resnet_degradation | 2 -->
## 1. resnet_degradation (paraphrase, papers), run 2

**Question:** why very deep networks get worse and how learning a residual fixes it

**Expected answer:** Degradation problem; layers learn a residual mapping via shortcut connections

**Reference passage (what the paper says):**
> we address the degradation problem by introducing a deep residual learning framework

**Model's answer:**

> Very deep networks can get worse due to an optimization difficulty where the training error increases with depth, and this is more pronounced in plain networks that simply stack layers (Source [4][6]). This issue is not caused by vanishing gradients, as networks with batch normalization (BN) still achieve competitive accuracy with shallower networks (Source [4]). The problem seems to be related to the solver having exponentially low convergence rates (Source [4]). Learning a residual mapping instead of directly fitting a deep network helps address this issue. By reformulating the network to learn the residual function \( F(x) = H(x) - x \) instead of \( H(x) \), the optimization becomes easier. This is because the identity mapping, which is a solution by construction, ensures that the deeper model does not have higher training error than the shallower counterpart (Source [3]). Experiments show that residual networks can achieve better training and validation performance, even with increased depth (Source [4][6]).

**Sources the answer cites:**

**[3] Deep Residual Learning for Image Recognition (arxiv:1512.03385) §1 Introduction, pp.1-2**
> Deep convolutional neural networks [22,21] have led to a series of breakthroughs for image classification [21,50,40]. Deep networks naturally integrate low/mid/highlevel features [50] and classifiers in an end-to-end multilayer fashion, and the "levels" of features can be enriched by the number of stacked layers (depth). Recent evidence [41,44] reveals that network depth is of crucial importance, and the leading results [41,44,13,16] on the challenging ImageNet dataset [36] all exploit "very deep" [41] models, with a depth of sixteen [41] to thirty [16]. Many other nontrivial visual recognition tasks [8,12,7,32,27] have also 1 http://image-net.org/challenges/LSVRC/2015/ and http://mscoco.org/dataset/#detections-challenge2015. greatly benefited from very deep models. Driven by the significance of depth, a question arises: Is learning better networks as easy as stacking more layers? An obstacle to answering this question was the notorious problem of vanishing/exploding gradients [1,9], which hamper convergence from the beginning. This problem, however, has been largely addressed by normalized initialization [23,9,37,13] and intermediate normalization layers [16], which enable networks with tens of layers to start converging for stochastic gradient descent (SGD) with backpropagation [22]. When deeper networks are able to start converging, a degradation problem has been exposed: with the network depth increasing, accuracy gets saturated (which might be unsurprising) and then degrades rapidly. Unexpectedly, such degradation is not caused by overfitting, and adding more layers to a suitably deep model leads to higher training error, as reported in [11,42] and thoroughly verified by our experiments. Fig. 1 shows a typical example. The degradation (of training accuracy) indicates that not all systems are similarly easy to optimize. Let us consider a shallower architecture and its deeper counterpart that adds more layers onto it. There exists a solution by construction to the deeper model: the added layers are identity mapping, and the other layers are copied from the learned shallower model. The existence of this constructed solution indicates that a deeper model should produce no higher training error than its shallower counterpart. But experiments show that our current solvers on hand are unable to find solutions that are comparably good or better than the constructed solution (or unable to do so in feasible time). In this paper, we address the degradation problem by introducing a deep residual learning framework. Instead of hoping each few stacked layers directly fit a desired underlying mapping, we explicitly let these layers fit a residual mapping. Formally, denoting the desired underlying mapping as H(x), we let the stacked nonlinear layers fit another mapping of F(x) := H(x)x. The original mapping is recast into F(x)+x. We hypothesize that it is easier to optimize the residual mapping than to optimize the original, unreferenced mapping. To the extreme, if an identity mapping were optimal, it would be easier to push the residual to zero than to fit an identity mapping by a stack of nonlinear layers. The formulation of F(x) + x can be realized by feedforward neural networks with "shortcut connections" (Fig. 2). Shortcut connections [2,34,49] are those skipping one or more layers. In our case, the shortcut connections simply perform identity mapping, and their outputs are added to the outputs of the stacked layers (Fig. 2). Identity shortcut connections add neither extra parameter nor computational complexity. The entire network can still be trained end-to-end by SGD with backpropagation, and can be easily implemented using common libraries (e.g., Caffe [19]) without modifying the solvers. We present comprehensive experiments on ImageNet [36] to show the degradation problem and evaluate our method. We show that: 1) Our extremely deep residual nets are easy to optimize, but the counterpart "plain" nets (that simply stack layers) exhibit higher training error when the depth increases; 2) Our deep residual nets can easily enjoy accuracy gains from greatly increased depth, producing results substantially better than previous networks. Similar phenomena are also shown on the CIFAR-10 set [20], suggesting that the optimization difficulties and the effects of our method are not just akin to a particular dataset. We present successfully trained models on this dataset with over 100 layers, and explore models with over 1000 layers. On the ImageNet classification dataset [36], we obtain excellent results by extremely deep residual nets. Our 152layer residual net is the deepest network ever presented on ImageNet, while still having lower complexity than VGG nets [41]. Our ensemble has 3.57% top-5 error on the ImageNet test set, and won the 1st place in the ILSVRC 2015 classification competition. The extremely deep representations also have excellent generalization performance on other recognition tasks, and lead us to further win the 1st places on: ImageNet detection, ImageNet localization, COCO detection, and COCO segmentation in ILSVRC & COCO 2015 competitions. This strong evidence shows that the residual learning principle is generic, and we expect that it is applicable in other vision and non-vision problems.

**[4] Deep Residual Learning for Image Recognition (arxiv:1512.03385) §4 Experiments > 4.1 ImageNet Classification, p.5**
> We argue that this optimization difficulty is unlikely to be caused by vanishing gradients. These plain networks are trained with BN [16], which ensures forward propagated signals to have non-zero variances. We also verify that the backward propagated gradients exhibit healthy norms with BN. So neither forward nor backward signals vanish. In fact, the 34-layer plain net is still able to achieve competitive accuracy (Table 3), suggesting that the solver works to some extent. We conjecture that the deep plain nets may have exponentially low convergence rates, which impact the reducing of the training error 3 . The reason for such optimization difficulties will be studied in the future. Residual Networks. Next we evaluate 18-layer and 34layer residual nets (ResNets). The baseline architectures are the same as the above plain nets, expect that a shortcut connection is added to each pair of 3×3 filters as in Fig. 3 (right). In the first comparison (Table 2 and Fig. 4 right), we use identity mapping for all shortcuts and zero-padding for increasing dimensions (option A). So they have no extra parameter compared to the plain counterparts. We have three major observations from Table 2 and Fig. 4. First, the situation is reversed with residual learning -the 34-layer ResNet is better than the 18-layer ResNet (by 2.8%). More importantly, the 34-layer ResNet exhibits considerably lower training error and is generalizable to the validation data. This indicates that the degradation problem is well addressed in this setting and we manage to obtain accuracy gains from increased depth.

**[6] Deep Residual Learning for Image Recognition (arxiv:1512.03385) §4 Experiments > 4.2 CIFAR-10 and Analysis, pp.7-8**
> We compare n = {3, 5, 7, 9}, leading to 20, 32, 44, and 56-layer networks. Fig. 6 (left) shows the behaviors of the plain nets. The deep plain nets suffer from increased depth, and exhibit higher training error when going deeper. This phenomenon is similar to that on ImageNet (Fig. 4, left) and on MNIST (see [42]), suggesting that such an optimization difficulty is a fundamental problem. Fig. 6 (middle) shows the behaviors of ResNets. Also similar to the ImageNet cases (Fig. 4, right), our ResNets manage to overcome the optimization difficulty and demonstrate accuracy gains when the depth increases. We further explore n = 18 that leads to a 110-layer ResNet. In this case, we find that the initial learning rate of 0.1 is slightly too large to start converging 5 . So we use 0.01 to warm up the training until the training error is below 80% (about 400 iterations), and then go back to 0.1 and continue training. The rest of the learning schedule is as done previously. This 110-layer network converges well (Fig. 6, middle). It has fewer parameters than other deep and thin networks such as FitNet [35] and Highway [42] (Table 6), yet is among the state-of-the-art results (6.43%, Table 6).

verdict: correct
[3] supports: yes
[4] supports: yes
[6] supports: yes
note: explains both parts: training error rises with depth in plain nets (the degradation problem, not named) and residual reformulation F(x)=H(x)-x with the identity-mapping argument; two residual sentences carry no citation but [3] contains them

---

<!-- entry: papers-20261123T062859Z-fused | adam_adamax_bound | 1 -->
## 2. adam_adamax_bound (concept, papers), run 1

**Question:** Which Adam variant gives a simpler bound on the size of each parameter update?

**Expected answer:** AdaMax: |Δt| ≤ α

**Reference passage (what the paper says):**
> the magnitude of parameter updates has a simpler bound with AdaMax than Adam

**Model's answer:**

> AdaMax gives a simpler bound on the size of each parameter update, specifically |∆ t | ≤ α [2].

**Sources the answer cites:**

**[2] Adam: A Method for Stochastic Optimization (arxiv:1412.6980) §7 EXTENSIONS > 7.1 ADAMAX, pp.8-9**
> In Adam, the update rule for individual weights is to scale their gradients inversely proportional to a (scaled) L 2 norm of their individual current and past gradients. We can generalize the L 2 norm based update rule to a L p norm based update rule. Such variants become numerically unstable for large p. However, in the special case where we let p → ∞, a surprisingly simple and stable algorithm emerges; see algorithm 2. We'll now derive the algorithm. Let, in case of the L p norm, the stepsize at time t be inversely proportional to v 1/p t , where: v t = β p 2 v t-1 + (1 -β p 2 )|g t | p (6) = (1 -β p 2 ) t i=1 β p(t-i) 2 • |g i | p (7) Algorithm 2: AdaMax, a variant of Adam based on the infinity norm. See section 7.1 for details. Good default settings for the tested machine learning problems are α = 0.002, β 1 = 0.9 and β 2 = 0.999. With β t 1 we denote β 1 to the power t. Here, (α/(1β t 1 )) is the learning rate with the bias-correction term for the first moment. All operations on vectors are element-wise. Require: α: Stepsize Require: β 1 , β 2 ∈ [0, 1): Exponential decay rates Require: f (θ): Stochastic objective function with parameters θ Require: θ 0 : Initial parameter vector m 0 ← 0 (Initialize 1 st moment vector) u 0 ← 0 (Initialize the exponentially weighted infinity norm) t ← 0 (Initialize timestep) while θ t not converged do t ← t + 1 g t ← ∇ θ f t (θ t-1 ) (Get gradients w.r.t. stochastic objective at timestep t) m t ← β 1 • m t-1 + (1 -β 1 ) • g t (Update biased first moment estimate) u t ← max(β 2 • u t-1 , |g t |) (Update the exponentially weighted infinity norm) θ t ← θ t-1 -(α/(1 -β t 1 )) • m t /u t (Update parameters) end while return θ t (Resulting parameters) Note that the decay term is here equivalently parameterised as β p 2 instead of β 2 . Now let p → ∞, and define u t = lim p→∞ (v t ) 1/p , then: u t = lim p→∞ (v t ) 1/p = lim p→∞ (1 -β p 2 ) t i=1 β p(t-i) 2 • |g i | p 1/p (8) = lim p→∞ (1 -β p 2 ) 1/p t i=1 β p(t-i) 2 • |g i | p 1/p (9) = lim p→∞ t i=1 β (t-i) 2 • |g i | p 1/p (10) = max β t-1 2 |g 1 |, β t-2 2 |g 2 |, . . . , β 2 |g t-1 |, |g t | (11) Which corresponds to the remarkably simple recursive formula: u t = max(β 2 • u t-1 , |g t |) (12) with initial value u 0 = 0. Note that, conveniently enough, we don't need to correct for initialization bias in this case. Also note that the magnitude of parameter updates has a simpler bound with AdaMax than Adam, namely: |∆ t | ≤ α.

verdict: correct
[2] supports: yes
note: |Δt| ≤ α is in [2] (the end of the source; the first sheet cut it off)

---

<!-- entry: papers-20261123T062859Z-fused | none_alexnet_optimizer | 1 -->
## 3. none_alexnet_optimizer (unanswerable, papers), run 1

**Question:** Which optimizer and learning rate did AlexNet use?

**Expected answer:** *not in the corpus*: the right behaviour is to say the sources do not cover it.

**Model's answer:**

> The sources provided do not contain specific information about the optimizer and learning rate used by AlexNet. [2][3][4][5][6][7][8] focus on Adam optimizer and its variants, but do not mention AlexNet. Therefore, I cannot provide the answer to which optimizer and learning rate AlexNet used based on the given sources. [1][2] discuss the Adam optimizer parameters but do not specify their use in AlexNet.

**Sources the answer cites:**

**[1] Attention Is All You Need (arxiv:1706.03762) §5 Training > 5.3 Optimizer, p.7**
> We used the Adam optimizer [20] with β 1 = 0.9, β 2 = 0.98 and ϵ = 10 -9 . We varied the learning rate over the course of training, according to the formula: lrate = d -0.5 model • min(step_num -0.5 , step_num • warmup_steps -1.5 ) (3) This corresponds to increasing the learning rate linearly for the first warmup_steps training steps, and decreasing it thereafter proportionally to the inverse square root of the step number. We used warmup_steps = 4000.

**[2] Adam: A Method for Stochastic Optimization (arxiv:1412.6980) §6 EXPERIMENTS, p.5**
> To empirically evaluate the proposed method, we investigated different popular machine learning models, including logistic regression, multilayer fully connected neural networks and deep convolutional neural networks. Using large models and datasets, we demonstrate Adam can efficiently solve practical deep learning problems. We use the same parameter initialization when comparing different optimization algorithms. The hyper-parameters, such as learning rate and momentum, are searched over a dense grid and the results are reported using the best hyper-parameter setting.

**[3] Efficient Estimation of Word Representations in Vector Space (arxiv:1301.3781) §2 Model Architectures > 2.3 Parallel Training of Neural Networks, p.4**
> To train models on huge data sets, we have implemented several models on top of a large-scale distributed framework called DistBelief [6], including the feedforward NNLM and the new models proposed in this paper. The framework allows us to run multiple replicas of the same model in parallel, and each replica synchronizes its gradient updates through a centralized server that keeps all the parameters. For this parallel training, we use mini-batch asynchronous gradient descent with an adaptive learning rate procedure called Adagrad [7]. Under this framework, it is common to use one hundred or more model replicas, each using many CPU cores at different machines in a data center.

**[4] Adam: A Method for Stochastic Optimization (arxiv:1412.6980) §7 EXTENSIONS > 7.1 ADAMAX, p.9**
> Algorithm 2: AdaMax, a variant of Adam based on the infinity norm. See section 7.1 for details. Good default settings for the tested machine learning problems are α = 0.002, β 1 = 0.9 and β 2 = 0.999. With β t 1 we denote β 1 to the power t. Here, (α/(1β t 1 )) is the learning rate with the bias-correction term for the first moment. All operations on vectors are element-wise. Require: α: Stepsize Require: β 1 , β 2 ∈ [0, 1): Exponential decay rates Require: f (θ): Stochastic objective function with parameters θ Require: θ 0 : Initial parameter vector m 0 ← 0 (Initialize 1 st moment vector) u 0 ← 0 (Initialize the exponentially weighted infinity norm) t ← 0 (Initialize timestep) while θ t not converged do t ← t + 1 g t ← ∇ θ f t (θ t-1 ) (Get gradients w.r.t. stochastic objective at timestep t) m t ← β 1 • m t-1 + (1 -β 1 ) • g t (Update biased first moment estimate) u t ← max(β 2 • u t-1 , |g t |) (Update the exponentially weighted infinity norm) θ t ← θ t-1 -(α/(1 -β t 1 )) • m t /u t (Update parameters) end while return θ t (Resulting parameters)

**[5] Distilling the Knowledge in a Neural Network (arxiv:1503.02531) §1 Introduction, p.2**
> It is generally accepted that the objective function used for training should reflect the true objective of the user as closely as possible. Despite this, models are usually trained to optimize performance on the training data when the real objective is to generalize well to new data. It would clearly be better to train models to generalize well, but this requires information about the correct way to generalize and this information is not normally available. When we are distilling the knowledge from a large model into a small one, however, we can train the small model to generalize in the same way as the large model. If the cumbersome model generalizes well because, for example, it is the average of a large ensemble of different models, a small model trained to generalize in the same way will typically do much better on test data than a small model that is trained in the normal way on the same training set as was used to train the ensemble. An obvious way to transfer the generalization ability of the cumbersome model to a small model is to use the class probabilities produced by the cumbersome model as "soft targets" for training the small model. For this transfer stage, we could use the same training set or a separate "transfer" set. When the cumbersome model is a large ensemble of simpler models, we can use an arithmetic or geometric mean of their individual predictive distributions as the soft targets. When the soft targets have high entropy, they provide much more information per training case than hard targets and much less variance in the gradient between training cases, so the small model can often be trained on much less data than the original cumbersome model and using a much higher learning rate.

**[6] Adam: A Method for Stochastic Optimization (arxiv:1412.6980) §Figures and Tables, p.7**
> Figure 2: Training of multilayer neural networks on MNIST images. (a) Neural networks using dropout stochastic regularization. (b) Neural networks with deterministic cost function.We compare with the sum-of-functions (SFO) optimizer(Sohl-Dickstein et al., 2014)

**[7] Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift (arxiv:1502.03167) §1 Introduction, pp.1-2**
> Using mini-batches of examples, as opposed to one example at a time, is helpful in several ways. First, the gradient of the loss over a mini-batch is an estimate of the gradient over the training set, whose quality improves as the batch size increases. Second, computation over a batch can be much more efficient than m computations for individual examples, due to the parallelism afforded by the modern computing platforms. While stochastic gradient is simple and effective, it requires careful tuning of the model hyper-parameters, specifically the learning rate used in optimization, as well as the initial values for the model parameters. The training is complicated by the fact that the inputs to each layer are affected by the parameters of all preceding layers -so that small changes to the network parameters amplify as the network becomes deeper. The change in the distributions of layers' inputs presents a problem because the layers need to continuously adapt to the new distribution. When the input distribution to a learning system changes, it is said to experience covariate shift (Shimodaira, 2000). This is typically handled via domain adaptation (Jiang, 2008). However, the notion of covariate shift can be extended beyond the learning system as a whole, to apply to its parts, such as a sub-network or a layer. Consider a network computing ℓ = F 2 (F 1 (u, Θ 1 ), Θ 2 ) where F 1 and F 2 are arbitrary transformations, and the parameters Θ 1 , Θ 2 are to be learned so as to minimize the loss ℓ. Learning Θ 2 can be viewed as if the inputs x = F 1 (u, Θ 1 ) are fed into the sub-network ℓ = F 2 (x, Θ 2 ). For example, a gradient descent step Θ 2 ← Θ 2 - α m m i=1 ∂F 2 (x i , Θ 2 ) ∂Θ 2 (for batch size m and learning rate α) is exactly equivalent to that for a stand-alone network F 2 with input x. Therefore, the input distribution properties that make training more efficient -such as having the same distribution between the training and test data -apply to training the sub-network as well. As such it is advantageous for the distribution of x to remain fixed over time. Then, Θ 2 does not have to readjust to compensate for the change in the distribution of x. Fixed distribution of inputs to a sub-network would have positive consequences for the layers outside the subnetwork, as well. Consider a layer with a sigmoid activation function z = g(W u + b) where u is the layer input, the weight matrix W and bias vector b are the layer parameters to be learned, and g(x) = 1 1+exp(-x) . As |x| increases, g ′ (x) tends to zero. This means that for all dimensions of x = W u+b except those with small absolute values, the gradient flowing down to u will vanish and the model will train slowly. However, since x is affected by W, b and the parameters of all the layers below, changes to those parameters during training will likely move many dimensions of x into the saturated regime of the nonlinearity and slow down the convergence. This effect is amplified as the network depth increases. In practice, the saturation problem and the resulting vanishing gradients are usually addressed by using Rectified Linear Units (Nair & Hinton, 2010) ReLU (x) = max(x, 0), careful initialization (Bengio & Glorot, 2010;Saxe et al., 2013), and small learning rates. If, however, we could ensure that the distribution of nonlinearity inputs remains more stable as the network trains, then the optimizer would be less likely to get stuck in the saturated regime, and the training would accelerate.

**[8] Adam: A Method for Stochastic Optimization (arxiv:1412.6980) §4 CONVERGENCE ANALYSIS, p.4**
> Our following theorem holds when the learning rate α t is decaying at a rate of t -1 2 and first moment running average coefficient β 1,t decay exponentially with λ, that is typically close to 1, e.g. 1 -10 -8 . Theorem 4.1. Assume that the function f t has bounded gradients, ∇f t (θ) 2 ≤ G, ∇f t (θ) ∞ ≤ G ∞ for all θ ∈ R d and distance between any θ t generated by Adam is bounded, θ n -θ m 2 ≤ D, θ m -θ n ∞ ≤ D ∞ for any m, n ∈ {1, ..., T }, and β 1 , β 2 ∈ [0, 1) satisfy β 2 1 √ β2 < 1. Let α t = α √ t and β 1,t = β 1 λ t-1 , λ ∈ (0, 1). Adam achieves the following guarantee, for all T ≥ 1. R(T ) ≤ D 2 2α(1 -β 1 ) d i=1 T v T,i + α(1 + β 1 )G ∞ (1 -β 1 ) √ 1 -β 2 (1 -γ) 2 d i=1 g 1:T,i 2 + d i=1 D 2 ∞ G ∞ √ 1 -β 2 2α(1 -β 1 )(1 -λ) 2 Our Theorem 4.1 implies when the data features are sparse and bounded gradients, the summation term can be much smaller than its upper bound

verdict: refused_ok
[1] supports: yes
[2] supports: partly
[3] supports: no
[4] supports: yes
[5] supports: no
[6] supports: partly
[7] supports: no
[8] supports: yes
note: right refusal, but it cites all 8 sources and claims [2]-[8] 'focus on Adam': false for [3] word2vec/Adagrad, [5] distillation, [7] BatchNorm

---

<!-- entry: papers-20261123T062859Z-fused | resnet_ensemble_error | 1 -->
## 4. resnet_ensemble_error (identifier, papers), run 1

**Question:** What top-5 error did the ResNet ensemble reach on the ImageNet test set?

**Expected answer:** 3.57%

**Reference passage (what the paper says):**
> Our ensemble has 3.57% top-5 error on the ImageNet test set

**Model's answer:**

> Our ResNet ensemble achieved a top-5 validation error of 4.9% and a top-5 test error of 4.82% [3].

**Sources the answer cites:**

**[3] Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift (arxiv:1502.03167) §4 Experiments > 4.2 ImageNet classification > 4.2.3 Ensemble Classification, p.7**
> The current reported best results on the ImageNet Large Scale Visual Recognition Competition are reached by the Deep Image ensemble of traditional models (Wu et al., 2015) and the ensemble model of (He et al., 2015). The latter reports the top-5 error of 4.94%, as evaluated by the ILSVRC server. Here we report a top-5 validation error of 4.9%, and test error of 4.82% (according to the ILSVRC server). This improves upon the previous best result, and exceeds the estimated accuracy of human raters according to (Russakovsky et al., 2014). For our ensemble, we used 6 networks. Each was based on BN-x30, modified via some of the following: increased initial weights in the convolutional layers; using Dropout (with the Dropout probability of 5% or 10%, vs. 40% for the original Inception); and using non-convolutional, per-activation Batch Normalization with last hidden layers of the model. Each network achieved its maximum accuracy after about 6 • 10 6 training steps. The ensemble prediction was based on the arithmetic average of class probabilities predicted by the constituent networks. The details of ensemble and multicrop inference are similar to (Szegedy et al., 2014). We demonstrate in Fig. 4 that batch normalization allows us to set new state-of-the-art by a healthy margin on the ImageNet classification challenge benchmarks.

verdict: wrong
[3] supports: no
note: gives the BatchNorm paper's ensemble numbers (4.9% / 4.82%) and calls them ResNet's; expected 3.57%. ResNet's 3.57% chunk was not among the 8 sources (retrieval miss) and the model answered instead of saying so (generation failure); same in run 2

---

<!-- entry: papers-20261123T062859Z-fused | bn_covariate_shift | 1 -->
## 5. bn_covariate_shift (paraphrase, papers), run 1

**Question:** layer inputs keep changing distribution during training, which slows learning

**Expected answer:** Internal covariate shift

**Reference passage (what the paper says):**
> We define Internal Covariate Shift as the change in the distribution of network activations

**Model's answer:**

> [2] [3] The distribution of each layer's inputs changes during training as the parameters of the previous layers change. This phenomenon, referred to as internal covariate shift, slows down the training process by requiring lower learning rates and careful parameter initialization, and making it difficult to train models with saturating nonlinearities. [3]

**Sources the answer cites:**

**[2] Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift (arxiv:1502.03167) §1 Introduction, pp.1-2**
> Using mini-batches of examples, as opposed to one example at a time, is helpful in several ways. First, the gradient of the loss over a mini-batch is an estimate of the gradient over the training set, whose quality improves as the batch size increases. Second, computation over a batch can be much more efficient than m computations for individual examples, due to the parallelism afforded by the modern computing platforms. While stochastic gradient is simple and effective, it requires careful tuning of the model hyper-parameters, specifically the learning rate used in optimization, as well as the initial values for the model parameters. The training is complicated by the fact that the inputs to each layer are affected by the parameters of all preceding layers -so that small changes to the network parameters amplify as the network becomes deeper. The change in the distributions of layers' inputs presents a problem because the layers need to continuously adapt to the new distribution. When the input distribution to a learning system changes, it is said to experience covariate shift (Shimodaira, 2000). This is typically handled via domain adaptation (Jiang, 2008). However, the notion of covariate shift can be extended beyond the learning system as a whole, to apply to its parts, such as a sub-network or a layer. Consider a network computing ℓ = F 2 (F 1 (u, Θ 1 ), Θ 2 ) where F 1 and F 2 are arbitrary transformations, and the parameters Θ 1 , Θ 2 are to be learned so as to minimize the loss ℓ. Learning Θ 2 can be viewed as if the inputs x = F 1 (u, Θ 1 ) are fed into the sub-network ℓ = F 2 (x, Θ 2 ). For example, a gradient descent step Θ 2 ← Θ 2 - α m m i=1 ∂F 2 (x i , Θ 2 ) ∂Θ 2 (for batch size m and learning rate α) is exactly equivalent to that for a stand-alone network F 2 with input x. Therefore, the input distribution properties that make training more efficient -such as having the same distribution between the training and test data -apply to training the sub-network as well. As such it is advantageous for the distribution of x to remain fixed over time. Then, Θ 2 does not have to readjust to compensate for the change in the distribution of x. Fixed distribution of inputs to a sub-network would have positive consequences for the layers outside the subnetwork, as well. Consider a layer with a sigmoid activation function z = g(W u + b) where u is the layer input, the weight matrix W and bias vector b are the layer parameters to be learned, and g(x) = 1 1+exp(-x) . As |x| increases, g ′ (x) tends to zero. This means that for all dimensions of x = W u+b except those with small absolute values, the gradient flowing down to u will vanish and the model will train slowly. However, since x is affected by W, b and the parameters of all the layers below, changes to those parameters during training will likely move many dimensions of x into the saturated regime of the nonlinearity and slow down the convergence. This effect is amplified as the network depth increases. In practice, the saturation problem and the resulting vanishing gradients are usually addressed by using Rectified Linear Units (Nair & Hinton, 2010) ReLU (x) = max(x, 0), careful initialization (Bengio & Glorot, 2010;Saxe et al., 2013), and small learning rates. If, however, we could ensure that the distribution of nonlinearity inputs remains more stable as the network trains, then the optimizer would be less likely to get stuck in the saturated regime, and the training would accelerate.

**[3] Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift (arxiv:1502.03167) §Abstract, p.1**
> Training Deep Neural Networks is complicated by the fact that the distribution of each layer's inputs changes during training, as the parameters of the previous layers change. This slows down the training by requiring lower learning rates and careful parameter initialization, and makes it notoriously hard to train models with saturating nonlinearities. We refer to this phenomenon as internal covariate shift, and address the problem by normalizing layer inputs. Our method draws its strength from making normalization a part of the model architecture and performing the normalization for each training mini-batch. Batch Normalization allows us to use much higher learning rates and be less careful about initialization. It also acts as a regularizer, in some cases eliminating the need for Dropout. Applied to a state-of-the-art image classification model, Batch Normalization achieves the same accuracy with 14 times fewer training steps, and beats the original model by a significant margin. Using an ensemble of batchnormalized networks, we improve upon the best published result on ImageNet classification: reaching 4.9% top-5 validation error (and 4.8% test error), exceeding the accuracy of human raters.

verdict: correct
[2] supports: yes
[3] supports: yes
note: names internal covariate shift and its effect; citation markers oddly placed at the very start

---

<!-- entry: papers-20261123T062859Z-fused | none_bert_dropout | 1 -->
## 6. none_bert_dropout (unanswerable, papers), run 1

**Question:** What dropout rate does BERT use during pre-training?

**Expected answer:** *not in the corpus*: the right behaviour is to say the sources do not cover it.

**Model's answer:**

> The sources provided do not contain information about BERT or its dropout rate during pre-training. [4][5][6][7][8]

**Sources the answer cites:**

**[4] Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift (arxiv:1502.03167) §3 Normalization via Mini-Batch Statistics > 3.4 Batch Normalization regularizes the model, p.5**
> When training with Batch Normalization, a training example is seen in conjunction with other examples in the mini-batch, and the training network no longer producing deterministic values for a given training example. In our experiments, we found this effect to be advantageous to the generalization of the network. Whereas Dropout (Srivastava et al., 2014) is typically used to reduce overfitting, in a batch-normalized network we found that it can be either removed or reduced in strength.

**[5] Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift (arxiv:1502.03167) §4 Experiments > 4.2 ImageNet classification > 4.2.1 Accelerating BN Networks, p.6**
> Simply adding Batch Normalization to a network does not take full advantage of our method. To do so, we further changed the network and its training parameters, as follows: Increase learning rate. In a batch-normalized model, we have been able to achieve a training speedup from higher learning rates, with no ill side effects (Sec. 3.3). Remove Dropout. As described in Sec. 3.4, Batch Normalization fulfills some of the same goals as Dropout. Removing Dropout from Modified BN-Inception speeds up training, without increasing overfitting. Reduce the L 2 weight regularization. While in Inception an L 2 loss on the model parameters controls overfitting, in Modified BN-Inception the weight of this loss is reduced by a factor of 5. We find that this improves the accuracy on the held-out validation data. Accelerate the learning rate decay. In training Inception, learning rate was decayed exponentially. Because our network trains faster than Inception, we lower the learning rate 6 times faster. Remove Local Response Normalization While Inception and other networks (Srivastava et al., 2014) benefit from it, we found that with Batch Normalization it is not necessary. Shuffle training examples more thoroughly. We enabled within-shard shuffling of the training data, which prevents the same examples from always appearing in a mini-batch together. This led to about 1% improvements in the validation accuracy, which is consistent with the view of Batch Normalization as a regularizer (Sec. 3.4): the randomization inherent in our method should be most beneficial when it affects an example differently each time it is seen.

**[6] Attention Is All You Need (arxiv:1706.03762) §6.1 Machine Translation, p.8**
> On the WMT 2014 English-to-German translation task, the big transformer model (Transformer (big) in Table 2) outperforms the best previously reported models (including ensembles) by more than 2.0 BLEU, establishing a new state-of-the-art BLEU score of 28.4. The configuration of this model is listed in the bottom line of Table 3. Training took 3.5 days on 8 P100 GPUs. Even our base model surpasses all previously published models and ensembles, at a fraction of the training cost of any of the competitive models. On the WMT 2014 English-to-French translation task, our big model achieves a BLEU score of 41.0, outperforming all of the previously published single models, at less than 1/4 the training cost of the previous state-of-the-art model. The Transformer (big) model trained for English-to-French used dropout rate P drop = 0.1, instead of 0.3. For the base models, we used a single model obtained by averaging the last 5 checkpoints, which were written at 10-minute intervals. For the big models, we averaged the last 20 checkpoints. We used beam search with a beam size of 4 and length penalty α = 0.6 [38]. These hyperparameters were chosen after experimentation on the development set. We set the maximum output length during inference to input length + 50, but terminate early when possible [38].

**[7] Deep Residual Learning for Image Recognition (arxiv:1512.03385) §Appendix > A Object Detection Baselines, p.10**
> For the usage of BN layers, after pre-training, we compute the BN statistics (means and variances) for each layer on the ImageNet training set. Then the BN layers are fixed during fine-tuning for object detection. As such, the BN layers become linear activations with constant offsets and scales, and BN statistics are not updated by fine-tuning. We fix the BN layers mainly for reducing memory consumption in Faster R-CNN training.

**[8] Generative Adversarial Networks (arxiv:1406.2661) §5 Experiments, pp.5-6**
> We trained adversarial nets an a range of datasets including MNIST [23], the Toronto Face Database (TFD) [28], and CIFAR-10 [21]. The generator nets used a mixture of rectifier linear activations [19,9] and sigmoid activations, while the discriminator net used maxout [10] activations. Dropout [17] was applied in training the discriminator net. While our theoretical framework permits the use of dropout and other noise at intermediate layers of the generator, we used noise as the input to only the bottommost layer of the generator network. We estimate probability of the test set data under p g by fitting a Gaussian Parzen window to the samples generated with G and reporting the log-likelihood under this distribution. The σ parameter Model MNIST TFD DBN [3] 138 ± 2 1909 ± 66 Stacked CAE [3] 121 ± 1.6 2110 ± 50 Deep GSN [6] 214 ± 1.1 1890 ± 29 Adversarial nets 225 ± 2 2057 ± 26 Table 1: Parzen window-based log-likelihood estimates. The reported numbers on MNIST are the mean loglikelihood of samples on test set, with the standard error of the mean computed across examples. On TFD, we computed the standard error across folds of the dataset, with a different σ chosen using the validation set of each fold. On TFD, σ was cross validated on each fold and mean log-likelihood on each fold were computed.

verdict: refused_ok
[4] supports: no
[5] supports: no
[6] supports: no
[7] supports: no
[8] supports: no
note: right refusal; the citations support nothing ([6] has the Transformer's dropout 0.1, a near-distractor the model rightly did not use)

---

<!-- entry: papers-20261123T062859Z-fused | attn_positional | 1 -->
## 7. attn_positional (concept, papers), run 1

**Question:** Why does the Transformer need positional encodings?

**Expected answer:** No recurrence or convolution, so order must be injected

**Reference passage (what the paper says):**
> Since our model contains no recurrence and no convolution, in order for the model to make use of the order of the sequence

**Model's answer:**

> The Transformer needs positional encodings to make use of the order of the sequence since the model contains no recurrence and no convolution. This is necessary for the model to attend to the relative or absolute position of the tokens in the sequence [1].

**Sources the answer cites:**

**[1] Attention Is All You Need (arxiv:1706.03762) §3 Model Architecture > 3.5 Positional Encoding, p.6**
> Since our model contains no recurrence and no convolution, in order for the model to make use of the order of the sequence, we must inject some information about the relative or absolute position of the tokens in the sequence. To this end, we add "positional encodings" to the input embeddings at the bottoms of the encoder and decoder stacks. The positional encodings have the same dimension d model as the embeddings, so that the two can be summed. There are many choices of positional encodings, learned and fixed [9]. In this work, we use sine and cosine functions of different frequencies: P E (pos,2i) = sin(pos/10000 2i/dmodel ) P E (pos,2i+1) = cos(pos/10000 2i/dmodel ) where pos is the position and i is the dimension. That is, each dimension of the positional encoding corresponds to a sinusoid. The wavelengths form a geometric progression from 2π to 10000 • 2π. We chose this function because we hypothesized it would allow the model to easily learn to attend by relative positions, since for any fixed offset k, P E pos+k can be represented as a linear function of P E pos . We also experimented with using learned positional embeddings [9] instead, and found that the two versions produced nearly identical results (see Table 3 row (E)). We chose the sinusoidal version because it may allow the model to extrapolate to sequence lengths longer than the ones encountered during training.

verdict: correct
[1] supports: yes
note: 

---

<!-- entry: papers-20261123T062859Z-fused | adam_adamax_defaults | 1 -->
## 8. adam_adamax_defaults (identifier, papers), run 1

**Question:** What default step size α is suggested for AdaMax?

**Expected answer:** α = 0.002

**Reference passage (what the paper says):**
> Good default settings for the tested machine learning problems are α = 0.002

**Model's answer:**

> The default step size α suggested for AdaMax is 0.002. [1]

**Sources the answer cites:**

**[1] Adam: A Method for Stochastic Optimization (arxiv:1412.6980) §7 EXTENSIONS > 7.1 ADAMAX, pp.8-9**
> In Adam, the update rule for individual weights is to scale their gradients inversely proportional to a (scaled) L 2 norm of their individual current and past gradients. We can generalize the L 2 norm based update rule to a L p norm based update rule. Such variants become numerically unstable for large p. However, in the special case where we let p → ∞, a surprisingly simple and stable algorithm emerges; see algorithm 2. We'll now derive the algorithm. Let, in case of the L p norm, the stepsize at time t be inversely proportional to v 1/p t , where: v t = β p 2 v t-1 + (1 -β p 2 )|g t | p (6) = (1 -β p 2 ) t i=1 β p(t-i) 2 • |g i | p (7) Algorithm 2: AdaMax, a variant of Adam based on the infinity norm. See section 7.1 for details. Good default settings for the tested machine learning problems are α = 0.002, β 1 = 0.9 and β 2 = 0.999. With β t 1 we denote β 1 to the power t. Here, (α/(1β t 1 )) is the learning rate with the bias-correction term for the first moment. All operations on vectors are element-wise. Require: α: Stepsize Require: β 1 , β 2 ∈ [0, 1): Exponential decay rates Require: f (θ): Stochastic objective function with parameters θ Require: θ 0 : Initial parameter vector m 0 ← 0 (Initialize 1 st moment vector) u 0 ← 0 (Initialize the exponentially weighted infinity norm) t ← 0 (Initialize timestep) while θ t not converged do t ← t + 1 g t ← ∇ θ f t (θ t-1 ) (Get gradients w.r.t. stochastic objective at timestep t) m t ← β 1 • m t-1 + (1 -β 1 ) • g t (Update biased first moment estimate) u t ← max(β 2 • u t-1 , |g t |) (Update the exponentially weighted infinity norm) θ t ← θ t-1 -(α/(1 -β t 1 )) • m t /u t (Update parameters) end while return θ t (Resulting parameters) Note that the decay term is here equivalently parameterised as β p 2 instead of β 2 . Now let p → ∞, and define u t = lim p→∞ (v t ) 1/p , then: u t = lim p→∞ (v t ) 1/p = lim p→∞ (1 -β p 2 ) t i=1 β p(t-i) 2 • |g i | p 1/p (8) = lim p→∞ (1 -β p 2 ) 1/p t i=1 β p(t-i) 2 • |g i | p 1/p (9) = lim p→∞ t i=1 β (t-i) 2 • |g i | p 1/p (10) = max β t-1 2 |g 1 |, β t-2 2 |g 2 |, . . . , β 2 |g t-1 |, |g t | (11) Which corresponds to the remarkably simple recursive formula: u t = max(β 2 • u t-1 , |g t |) (12) with initial value u 0 = 0. Note that, conveniently enough, we don't need to correct for initialization bias in this case. Also note that the magnitude of parameter updates has a simpler bound with AdaMax than Adam, namely: |∆ t | ≤ α.

verdict: correct
[1] supports: yes
note: 

---

<!-- entry: papers-20261123T062859Z-fused | none_gpt3_params | 1 -->
## 9. none_gpt3_params (unanswerable, papers), run 1

**Question:** How many parameters does GPT-3 have?

**Expected answer:** *not in the corpus*: the right behaviour is to say the sources do not cover it.

**Model's answer:**

> The sources provided do not contain information about the number of parameters in GPT-3. [1][2][3][4][5][6][7][8]

**Sources the answer cites:**

**[1] Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift (arxiv:1502.03167) §1 Introduction, p.2**
> Fixed distribution of inputs to a sub-network would have positive consequences for the layers outside the subnetwork, as well. Consider a layer with a sigmoid activation function z = g(W u + b) where u is the layer input, the weight matrix W and bias vector b are the layer parameters to be learned, and g(x) = 1 1+exp(-x) . As |x| increases, g ′ (x) tends to zero. This means that for all dimensions of x = W u+b except those with small absolute values, the gradient flowing down to u will vanish and the model will train slowly. However, since x is affected by W, b and the parameters of all the layers below, changes to those parameters during training will likely move many dimensions of x into the saturated regime of the nonlinearity and slow down the convergence. This effect is amplified as the network depth increases. In practice, the saturation problem and the resulting vanishing gradients are usually addressed by using Rectified Linear Units (Nair & Hinton, 2010) ReLU (x) = max(x, 0), careful initialization (Bengio & Glorot, 2010;Saxe et al., 2013), and small learning rates. If, however, we could ensure that the distribution of nonlinearity inputs remains more stable as the network trains, then the optimizer would be less likely to get stuck in the saturated regime, and the training would accelerate.

**[2] Distilling the Knowledge in a Neural Network (arxiv:1503.02531) §5 Training ensembles of specialists on very big datasets > 5.1 The JFT dataset, pp.5-6**
> JFT is an internal Google dataset that has 100 million labeled images with 15,000 labels. When we did this work, Google's baseline model for JFT was a deep convolutional neural network [7] that had been trained for about six months using asynchronous stochastic gradient descent on a large number of cores. This training used two types of parallelism [2]. First, there were many replicas of the neural net running on different sets of cores and processing different mini-batches from the training set. Each replica computes the average gradient on its current mini-batch and sends this gradient to a sharded parameter server which sends back new values for the parameters. These new values reflect all of the gradients received by the parameter server since the last time it sent parameters to the replica. Second, each replica is spread over multiple cores by putting different subsets of the neurons on each core. Ensemble training is yet a third type of parallelism that can be wrapped JFT 1: Tea party; Easter; Bridal shower; Baby shower; Easter Bunny; ... JFT 2: Bridge; Cable-stayed bridge; Suspension bridge; Viaduct; Chimney; ... JFT 3: Toyota Corolla E100; Opel Signum; Opel Astra; Mazda Familia; ... Table 2: Example classes from clusters computed by our covariance matrix clustering algorithm around the other two types, but only if a lot more cores are available. Waiting for several years to train an ensemble of models was not an option, so we needed a much faster way to improve the baseline model.

**[3] Attention Is All You Need (arxiv:1706.03762) §6.2 Model Variations, pp.8-9**
> To evaluate the importance of different components of the Transformer, we varied our base model in different ways, measuring the change in performance on English-to-German translation on the . We used beam search as described in the previous section, but no checkpoint averaging. We present these results in Table 3. In Table 3 rows (A), we vary the number of attention heads and the attention key and value dimensions, keeping the amount of computation constant, as described in Section 3.2.2. While single-head attention is 0.9 BLEU worse than the best setting, quality also drops off with too many heads. In Table 3 rows (B), we observe that reducing the attention key size d k hurts model quality. This suggests that determining compatibility is not easy and that a more sophisticated compatibility function than dot product may be beneficial. We further observe in rows (C) and (D) that, as expected, bigger models are better, and dropout is very helpful in avoiding over-fitting. In row (E) we replace our sinusoidal positional encoding with learned positional embeddings [9], and observe nearly identical results to the base model.

**[4] Sequence to Sequence Learning with Neural Networks (arxiv:1409.3215) §3 Experiments > 3.4 Training details, p.5**
> After 5 epochs, we begun halving the learning rate every half epoch. We trained our models for a total of 7.5 epochs. • We used batches of 128 sequences for the gradient and divided it the size of the batch (namely, 128). • Although LSTMs tend to not suffer from the vanishing gradient problem, they can have exploding gradients. Thus we enforced a hard constraint on the norm of the gradient [10,25] by scaling it when its norm exceeded a threshold. For each training batch, we compute s = g 2 , where g is the gradient divided by 128. If s > 5, we set g = 5g s . • Different sentences have different lengths. Most sentences are short (e.g., length [20][21][22][23][24][25][26][27][28][29][30] but some sentences are long (e.g., length > 100), so a minibatch of 128 randomly chosen training sentences will have many short sentences and few long sentences, and as a result, much of the computation in the minibatch is wasted. To address this problem, we made sure that all sentences in a minibatch are roughly of the same length, yielding a 2x speedup.

**[5] Attention Is All You Need (arxiv:1706.03762) §3 Model Architecture > 3.3 Position-wise Feed-Forward Networks, p.5**
> In addition to attention sub-layers, each of the layers in our encoder and decoder contains a fully connected feed-forward network, which is applied to each position separately and identically. This consists of two linear transformations with a ReLU activation in between. FFN(x) = max(0, xW 1 + b 1 )W 2 + b 2 (2) While the linear transformations are the same across different positions, they use different parameters from layer to layer. Another way of describing this is as two convolutions with kernel size 1. The dimensionality of input and output is d model = 512, and the inner-layer has dimensionality d f f = 2048.

**[6] Denoising Diffusion Probabilistic Models (arxiv:2006.11239) §Appendix > B Experimental details, p.14**
> Our 32 × 32 models use four feature map resolutions (32 × 32 to 4 × 4), and our 256 × 256 models use six. All models have two convolutional residual blocks per resolution level and self-attention blocks at the 16 × 16 resolution between the convolutional blocks [6]. Diffusion time t is specified by adding the Transformer sinusoidal position embedding [60] into each residual block. Our CIFAR10 model has 35.7 million parameters, and our LSUN and CelebA-HQ models have 114 million parameters. We also trained a larger variant of the LSUN Bedroom model with approximately 256 million parameters by increasing filter count. We used TPU v3-8 (similar to 8 V100 GPUs) for all experiments. Our CIFAR model trains at 21 steps per second at batch size 128 (10.6 hours to train to completion at 800k steps), and sampling a batch of 256 images takes 17 seconds. Our CelebA-HQ/LSUN (256 2 ) models train at 2.2 steps per second at batch size 64, and sampling a batch of 128 images takes 300 seconds. We trained on CelebA-HQ for 0.5M steps, LSUN Bedroom for 2.4M steps, LSUN Cat for 1.8M steps, and LSUN Church for 1.2M steps. The larger LSUN Bedroom model was trained for 1.15M steps.

**[7] Auto-Encoding Variational Bayes (arxiv:1312.6114) §Appendix > C MLP's as probabilistic encoders and decoders > C.2 Gaussian MLP as encoder or decoder, p.11**
> In this case let encoder or decoder be a multivariate Gaussian with a diagonal covariance structure: where {W 3 , W 4 , W 5 , b 3 , b 4 , b 5 } are the weights and biases of the MLP and part of θ when used as decoder. Note that when this network is used as an encoder q φ (z|x), then z and x are swapped, and the weights and biases are variational parameters φ.

**[8] Attention Is All You Need (arxiv:1706.03762) §4 Why Self-Attention, p.6**
> In this section we compare various aspects of self-attention layers to the recurrent and convolutional layers commonly used for mapping one variable-length sequence of symbol representations (x 1 , ..., x n ) to another sequence of equal length (z 1 , ..., z n ), with x i , z i ∈ R d , such as a hidden layer in a typical sequence transduction encoder or decoder. Motivating our use of self-attention we consider three desiderata. One is the total computational complexity per layer. Another is the amount of computation that can be parallelized, as measured by the minimum number of sequential operations required. The third is the path length between long-range dependencies in the network. Learning long-range dependencies is a key challenge in many sequence transduction tasks. One key factor affecting the ability to learn such dependencies is the length of the paths forward and backward signals have to traverse in the network. The shorter these paths between any combination of positions in the input and output sequences, the easier it is to learn long-range dependencies [12]. Hence we also compare the maximum path length between any two input and output positions in networks composed of the different layer types.

verdict: refused_ok
[1] supports: no
[2] supports: no
[3] supports: no
[4] supports: no
[5] supports: no
[6] supports: no
[7] supports: no
[8] supports: no
note: right refusal; cites all 8 sources, none about GPT-3 ([6] has DDPM parameter counts, not used)

---

<!-- entry: papers-20261123T062859Z-fused | adam_bias_correction | 1 -->
## 10. adam_bias_correction (paraphrase, papers), run 1

**Question:** correcting moving averages of the gradient that start at zero

**Expected answer:** Bias-corrected first and second moment estimates

**Reference passage (what the paper says):**
> leading to moment estimates that are biased towards zero

**Model's answer:**

> [2] Adam's method corrects the moving averages of the gradient that start at zero by dividing by the term (1 - β²ᵗ) to counteract the initialization bias. This correction is necessary, especially when the decay rate β₂ is close to 1, to ensure that the moment estimates are not biased towards zero, especially during the initial timesteps. [2]

**Sources the answer cites:**

**[2] Adam: A Method for Stochastic Optimization (arxiv:1412.6980) §3 INITIALIZATION BIAS CORRECTION, p.3**
> As explained in section 2, Adam utilizes initialization bias correction terms. We will here derive the term for the second moment estimate; the derivation for the first moment estimate is completely analogous. Let g be the gradient of the stochastic objective f , and we wish to estimate its second raw moment (uncentered variance) using an exponential moving average of the squared gradient, with decay rate β 2 . Let g 1 , ..., g T be the gradients at subsequent timesteps, each a draw from an underlying gradient distribution g t ∼ p(g t ). Let us initialize the exponential moving average as v 0 = 0 (a vector of zeros). First note that the update at timestep t of the exponential moving average v t = β 2 • v t-1 + (1 -β 2 ) • g 2 t ( where g 2 t indicates the elementwise square g t g t ) can be written as a function of the gradients at all previous timesteps: v t = (1 -β 2 ) t i=1 β t-i 2 • g 2 i (1) We wish to know how E[v t ], the expected value of the exponential moving average at timestep t, relates to the true second moment E[g 2 t ], so we can correct for the discrepancy between the two. Taking expectations of the left-hand and right-hand sides of eq. ( 1): E[v t ] = E (1 -β 2 ) t i=1 β t-i 2 • g 2 i (2) = E[g 2 t ] • (1 -β 2 ) t i=1 β t-i 2 + ζ (3) = E[g 2 t ] • (1 -β t 2 ) + ζ (4) where ζ = 0 if the true second moment E[g 2 i ] is stationary; otherwise ζ can be kept small since the exponential decay rate β 1 can (and should) be chosen such that the exponential moving average assigns small weights to gradients too far in the past. What is left is the term (1β t 2 ) which is caused by initializing the running average with zeros. In algorithm 1 we therefore divide by this term to correct the initialization bias. In case of sparse gradients, for a reliable estimate of the second moment one needs to average over many gradients by chosing a small value of β 2 ; however it is exactly this case of small β 2 where a lack of initialisation bias correction would lead to initial steps that are much larger.

verdict: correct
[2] supports: yes
note: describes the second-moment correction only (dividing by 1 - β2^t, written ambiguously as β²ᵗ); the question did not ask for both moments

---

<!-- entry: papers-20261123T062859Z-fused | bn_gamma_beta | 1 -->
## 11. bn_gamma_beta (concept, papers), run 1

**Question:** Which learned parameters does batch normalization add for each activation, and what are they for?

**Expected answer:** γ and β, which scale and shift the normalized value

**Reference passage (what the paper says):**
> which scale and shift the normalized value

**Model's answer:**

> Batch normalization adds two learned parameters for each activation: a scaling parameter γ and a shifting parameter β. These parameters are used to scale and shift the normalized value, as shown in the formula: y (k) = γ (k) x (k) + β (k) [1]

**Sources the answer cites:**

**[1] Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift (arxiv:1502.03167) §3 Normalization via Mini-Batch Statistics, pp.3-4**
> Since the full whitening of each layer's inputs is costly and not everywhere differentiable, we make two necessary simplifications. The first is that instead of whitening the features in layer inputs and outputs jointly, we will normalize each scalar feature independently, by making it have the mean of zero and the variance of 1. For a layer with d-dimensional input x = (x (1) . . . x (d) ), we will normalize each dimension x (k) = x (k) -E[x (k) ] Var[x (k) ] where the expectation and variance are computed over the training data set. As shown in (LeCun et al., 1998b), such normalization speeds up convergence, even when the features are not decorrelated. Note that simply normalizing each input of a layer may change what the layer can represent. For instance, normalizing the inputs of a sigmoid would constrain them to the linear regime of the nonlinearity. To address this, we make sure that the transformation inserted in the network can represent the identity transform. To accomplish this, we introduce, for each activation x (k) , a pair of parameters γ (k) , β (k) , which scale and shift the normalized value: y (k) = γ (k) x (k) + β (k) . These parameters are learned along with the original model parameters, and restore the representation power of the network. Indeed, by setting γ (k) = Var[x (k) ] and β (k) = E[x (k) ], we could recover the original activations, if that were the optimal thing to do. In the batch setting where each training step is based on the entire training set, we would use the whole set to normalize activations. However, this is impractical when using stochastic optimization. Therefore, we make the second simplification: since we use mini-batches in stochastic gradient training, each mini-batch produces estimates of the mean and variance of each activation. This way, the statistics used for normalization can fully participate in the gradient backpropagation. Note that the use of minibatches is enabled by computation of per-dimension variances rather than joint covariances; in the joint case, regularization would be required since the mini-batch size is likely to be smaller than the number of activations being whitened, resulting in singular covariance matrices. Consider a mini-batch B of size m. Since the normalization is applied to each activation independently, let us focus on a particular activation x (k) and omit k for clarity. We have m values of this activation in the mini-batch, B = {x 1...m }. Let the normalized values be x 1...m , and their linear transformations be y 1...m . We refer to the transform BN γ,β : x 1...m → y 1...m as the Batch Normalizing Transform. We present the BN Transform in Algorithm 1. In the algorithm, ǫ is a constant added to the mini-batch variance for numerical stability. Input: Values of x over a mini-batch: B = {x 1...m }; Parameters to be learned: γ, β Output: {y i = BN γ,β (x i )} µ B ← 1 m m i=1 x i // mini-batch mean σ 2 B ← 1 m m i=1 (x i -µ B ) 2 // mini-batch variance x i ← x i -µ B σ 2 B + ǫ // normalize y i ← γ x i + β ≡ BN γ,β (x i ) // Algorithm 1: Batch Normalizing Transform, applied to activation x over a mini-batch. The BN transform can be added to a network to manipulate any activation. In the notation y = BN γ,β (x), we indicate that the parameters γ and β are to be learned, but it should be noted that the BN transform does not independently process the activation in each training example. Rather, BN γ,β (x) depends both on the training example and the other examples in the mini-batch. The scaled and shifted values y are passed to other network layers. The normalized activations x are internal to our transformation, but their presence is crucial. The distributions of values of any x has the expected value of 0 and the variance of 1, as long as the elements of each mini-batch are sampled from the same distribution, and if we neglect ǫ. This can be seen by observing that m i=1 x i = 0 and 1 m m i=1 x 2 i = 1, and taking expectations. Each normalized activation x (k) can be viewed as an input to a sub-network composed of the linear transform y (k) = γ (k) x (k) + β (k) , followed by the other processing done by the original network. These sub-network inputs all have fixed means and variances, and although the joint distribution of these normalized x (k) can change over the course of training, we expect that the introduction of normalized inputs accelerates the training of the sub-network and, consequently, the network as a whole.

verdict: correct
[1] supports: yes
note: 

---

<!-- entry: hr-demo-20261123T063124Z-fused | handbook_paid_holidays | 1 -->
## 12. handbook_paid_holidays (missing_information, hr-demo), run 1

**Question:** How many paid company holidays are observed each year?

**Expected answer:** *not in the corpus*: the right behaviour is to say the sources do not cover it.

**Model's answer:**

> The Acme Corp Employee Handbook does not specify the number of paid company holidays observed each year. [1][2][3][4][5][6][7][8]

**Sources the answer cites:**

**[1] Acme Corp Employee Handbook (file:employee_handbook) §Section 6: Compensation and Benefits, p.?**
> Salary bands are reviewed annually in March. Merit increases, when granted, take effect in the first payroll of April. Overtime for non-exempt employees requires pre-approval and is paid at one and one half times the regular hourly rate for hours beyond 40 in a workweek. Full-time employees receive medical, dental, and vision coverage beginning on the first day of the month following hire. Acme Corp matches 401(k) contributions at 50 percent of employee contributions up to 6 percent of eligible compensation. Employees accrue 15 days of paid vacation in years one through three of service, 20 days in years four through seven, and 25 days thereafter. Sick leave accrues at one day per month up to a maximum bank of 30 days. Parental leave of 12 weeks paid is available to eligible employees after 12 months of continuous service.

**[2] Acme Corp Employee Handbook (file:employee_handbook) §Section 14: Travel and International Assignments, p.?**
> Domestic travel for customer meetings should be consolidated when multiple accounts are in the same metro area within a seven-day window. International travel requires VP approval and a completed travel risk assessment for destinations under a government advisory. Employees on temporary international assignment longer than 90 days must complete tax counseling with the global mobility vendor before departure. Travel insurance is provided automatically for approved trips booked through the corporate portal. Personal side trips attached to business travel are allowed if incremental costs are paid by the employee and documented separately. Export-control restricted technical documents must not be carried into countries on the company restricted-travel list without Legal clearance.

**[3] Acme Corp Employee Handbook (file:employee_handbook) §Section 11: Performance Management, p.?**
> Goals are set each quarter using the company OKR template. Managers hold 1:1 meetings at least biweekly. Formal performance reviews occur twice per year, in June and December. Employees rated Below Expectations receive a written performance improvement plan lasting 30 to 60 days with concrete milestones. Failure to meet the plan may result in role change or separation. Promotion packets require evidence of scope increase, sustained impact, and peer feedback from at least three collaborators. Calibration sessions across teams are required before final rating submission to reduce bias. Employees may submit a written rebuttal to HR within ten business days of receiving a formal review.

**[4] Acme Corp Employee Handbook (file:employee_handbook) §Section 3: Hybrid and Remote Work, p.?**
> Acme Corp operates a hybrid model. Roles designated as Office-Primary require at least three in-office days per week. Roles designated as Hybrid-Flexible require at least two in-office days per week. Fully remote roles are limited to positions approved by both the hiring manager and the department VP. Employees working remotely must maintain a private workspace suitable for confidential discussions, use company-managed devices for all production systems, and join video meetings with camera on unless bandwidth or accessibility constraints apply. Home office stipends of up to 400 dollars per calendar year may be reimbursed for chairs, monitors, or headsets with receipts submitted within 45 days of purchase. Acme Corp does not reimburse rent, utilities, or personal internet service.

**[5] Acme Corp Employee Handbook (file:employee_handbook) §Section 15: Procurement and Vendor Management, p.?**
> Purchases under 1,000 dollars may be made on a corporate card with manager approval. Purchases between 1,000 and 10,000 dollars require a purchase order. Purchases above 10,000 dollars require Legal review of the vendor contract and Security review if the vendor will process company or customer data. Preferred vendors listed in the procurement catalog should be used when they meet requirements. Employees must not accept personal gifts from vendors valued over 75 dollars. Annual vendor access reviews occur each November; unused vendor accounts are disabled. Shadow IT tools that store company data without IT approval are prohibited and may be blocked at the network layer.

**[6] Acme Corp Employee Handbook (file:employee_handbook) §Section 13: Health, Safety, and Facilities, p.?**
> Office locations maintain first-aid kits on every floor and clearly marked emergency exits. Fire drills occur twice per year. Employees who observe unsafe conditions must report them to Facilities the same day. Desk hoteling desks must be cleared by 6:00 p.m.; personal items left overnight may be moved to Lost and Found after 48 hours. Kitchen areas must be kept clean; food left unlabeled in shared refrigerators is discarded every Friday at 4:00 p.m. Bicycles may be stored only in designated racks. Visitors must sign in at reception and be escorted by an employee at all times in badge-restricted areas. After-hours building access requires an active badge and is logged automatically.

**[7] Acme Corp Employee Handbook (file:employee_handbook) §Section 4: Code of Conduct, p.?**
> Employees must treat colleagues, customers, and partners with respect. Harassment, discrimination, bullying, or retaliation of any kind is prohibited. This includes conduct on company premises, at company events, and on company communication channels. Romantic relationships between a manager and a direct report must be disclosed to HR immediately so reporting lines can be adjusted. Conflicts of interest, including outside employment with a competitor or ownership stakes exceeding 5 percent in a vendor company, must be disclosed in writing. Employees may not solicit coworkers for non-company commercial products during work hours. Violations are investigated by HR and may result in coaching, written warning, unpaid suspension, or termination depending on severity and prior history.

**[8] Acme Corp Employee Handbook (file:employee_handbook) §Section 2: Working Hours and Attendance, p.?**
> Standard office hours are Monday through Friday, 9:00 a.m. to 5:30 p.m. local time, with a 30-minute unpaid lunch break. Teams may adopt flexible start times between 7:30 a.m. and 10:00 a.m. with manager approval, provided core collaboration hours of 11:00 a.m. to 3:00 p.m. are covered. Remote employees must remain reachable on company chat during their scheduled hours. Repeated unexplained absences, or three late arrivals within a rolling 30-day window, trigger a written attendance review. Planned time off must be submitted in the HR portal at least five business days in advance for absences of one or two days, and at least fifteen business days in advance for absences of three or more consecutive days.

verdict: refused_ok
[1] supports: no
[2] supports: no
[3] supports: no
[4] supports: no
[5] supports: no
[6] supports: no
[7] supports: no
[8] supports: no
note: right refusal, worded as 'the Handbook does not specify'; cites all 8 sources, none about holidays

---

<!-- entry: hr-demo-20261123T063124Z-fused | handbook_late_arrivals | 1 -->
## 13. handbook_late_arrivals (direct, hr-demo), run 1

**Question:** How many late arrivals within a rolling 30-day window trigger an attendance review?

**Expected answer:** three late arrivals within a rolling 30-day window

**Reference passage (what the paper says):**
> three late arrivals within a rolling 30-day window

**Model's answer:**

> Three late arrivals within a rolling 30-day window trigger a written attendance review [1].

**Sources the answer cites:**

**[1] Acme Corp Employee Handbook (file:employee_handbook) §Section 2: Working Hours and Attendance, p.?**
> Standard office hours are Monday through Friday, 9:00 a.m. to 5:30 p.m. local time, with a 30-minute unpaid lunch break. Teams may adopt flexible start times between 7:30 a.m. and 10:00 a.m. with manager approval, provided core collaboration hours of 11:00 a.m. to 3:00 p.m. are covered. Remote employees must remain reachable on company chat during their scheduled hours. Repeated unexplained absences, or three late arrivals within a rolling 30-day window, trigger a written attendance review. Planned time off must be submitted in the HR portal at least five business days in advance for absences of one or two days, and at least fifteen business days in advance for absences of three or more consecutive days.

verdict: correct
[1] supports: yes
note: 

---

<!-- entry: hr-demo-20261123T063124Z-fused | handbook_equity_grant | 1 -->
## 14. handbook_equity_grant (missing_information, hr-demo), run 1

**Question:** Do employees receive stock options or RSUs?

**Expected answer:** *not in the corpus*: the right behaviour is to say the sources do not cover it.

**Model's answer:**

> The provided sources do not mention stock options or restricted stock units (RSUs). [1][2][3][4][5][6][7][8]

**Sources the answer cites:**

**[1] Acme Corp Employee Handbook (file:employee_handbook) §Section 6: Compensation and Benefits, p.?**
> Salary bands are reviewed annually in March. Merit increases, when granted, take effect in the first payroll of April. Overtime for non-exempt employees requires pre-approval and is paid at one and one half times the regular hourly rate for hours beyond 40 in a workweek. Full-time employees receive medical, dental, and vision coverage beginning on the first day of the month following hire. Acme Corp matches 401(k) contributions at 50 percent of employee contributions up to 6 percent of eligible compensation. Employees accrue 15 days of paid vacation in years one through three of service, 20 days in years four through seven, and 25 days thereafter. Sick leave accrues at one day per month up to a maximum bank of 30 days. Parental leave of 12 weeks paid is available to eligible employees after 12 months of continuous service.

**[2] Acme Corp Employee Handbook (file:employee_handbook) §Section 1: Employment Classification, p.?**
> Acme Corp classifies workers as full-time employees, part-time employees, temporary employees, or independent contractors. Full-time employees generally work at least 40 hours per week and are eligible for the complete benefits package described in Section 6. Part-time employees work fewer than 30 hours per week and receive a reduced benefits package. Temporary employees are hired for a defined project window, typically no longer than six months, and are not eligible for retirement matching. Independent contractors are not employees; they invoice through Accounts Payable and are responsible for their own taxes and insurance. Misclassification concerns should be reported to HR within five business days of discovery.

**[3] Acme Corp Employee Handbook (file:employee_handbook) §Section 5: Confidentiality and Intellectual Property, p.?**
> All product roadmaps, customer lists, pricing models, unpublished research, and internal financial forecasts are confidential. Employees may not share confidential materials outside Acme Corp without a signed non-disclosure agreement reviewed by Legal. Work product created within the scope of employment is owned by Acme Corp, including source code, designs, documents, and inventions. Employees must return all company devices and revoke personal copies of company data on or before their last working day. Personal cloud accounts and personal USB drives must not be used to store company source code or customer personal data. Suspected data leaks must be reported to security@acmecorp.example within one hour of discovery.

**[4] Acme Corp Employee Handbook (file:employee_handbook) §Section 11: Performance Management, p.?**
> Goals are set each quarter using the company OKR template. Managers hold 1:1 meetings at least biweekly. Formal performance reviews occur twice per year, in June and December. Employees rated Below Expectations receive a written performance improvement plan lasting 30 to 60 days with concrete milestones. Failure to meet the plan may result in role change or separation. Promotion packets require evidence of scope increase, sustained impact, and peer feedback from at least three collaborators. Calibration sessions across teams are required before final rating submission to reduce bias. Employees may submit a written rebuttal to HR within ten business days of receiving a formal review.

**[5] Acme Corp Employee Handbook (file:employee_handbook) §Section 17: Separation from Employment, p.?**
> Resignations should be submitted in writing with at least two weeks of notice for individual contributors and at least four weeks of notice for managers. Acme Corp may place employees on garden leave during the notice period when access to sensitive systems presents risk. Exit interviews are optional but encouraged. Final paychecks include accrued unused vacation where required by local law. Benefits coverage generally ends on the last day of employment; COBRA information is mailed within 14 days where applicable. Former employees must not retain access credentials, documents, or devices. Non-solicit and confidentiality obligations survive separation according to the employment agreement.

**[6] Acme Corp Employee Handbook (file:employee_handbook) §Section 4: Code of Conduct, p.?**
> Employees must treat colleagues, customers, and partners with respect. Harassment, discrimination, bullying, or retaliation of any kind is prohibited. This includes conduct on company premises, at company events, and on company communication channels. Romantic relationships between a manager and a direct report must be disclosed to HR immediately so reporting lines can be adjusted. Conflicts of interest, including outside employment with a competitor or ownership stakes exceeding 5 percent in a vendor company, must be disclosed in writing. Employees may not solicit coworkers for non-company commercial products during work hours. Violations are investigated by HR and may result in coaching, written warning, unpaid suspension, or termination depending on severity and prior history.

**[7] Acme Corp Employee Handbook (file:employee_handbook) §Section 15: Procurement and Vendor Management, p.?**
> Purchases under 1,000 dollars may be made on a corporate card with manager approval. Purchases between 1,000 and 10,000 dollars require a purchase order. Purchases above 10,000 dollars require Legal review of the vendor contract and Security review if the vendor will process company or customer data. Preferred vendors listed in the procurement catalog should be used when they meet requirements. Employees must not accept personal gifts from vendors valued over 75 dollars. Annual vendor access reviews occur each November; unused vendor accounts are disabled. Shadow IT tools that store company data without IT approval are prohibited and may be blocked at the network layer.

**[8] Acme Corp Employee Handbook (file:employee_handbook) §Section 8: Information Security, p.?**
> Employees must use unique passwords of at least 14 characters and enable multi-factor authentication on all company accounts. Passwords must not be shared, written on sticky notes, or stored in unencrypted personal notes apps. Company laptops must remain encrypted, locked when unattended, and patched within seven days of critical security updates. Public Wi-Fi may be used only through the company VPN. Phishing simulations occur quarterly; failing two consecutive simulations requires a short security refresher course. Software installation on managed devices requires approval through the IT service desk except for tools on the approved software catalog. Production credentials must never be committed to source control. Security incidents involving ransomware, account takeover, or lost devices must be reported immediately to the on-call security engineer listed in the internal directory.

verdict: refused_ok
[1] supports: no
[2] supports: no
[3] supports: no
[4] supports: no
[5] supports: no
[6] supports: no
[7] supports: no
[8] supports: no
note: right refusal; cites all 8 sources, none about equity

---

<!-- entry: hr-demo-20261123T063124Z-fused | policy_loyalty_25_and_expiry | 1 -->
## 15. policy_loyalty_25_and_expiry (multi_step, hr-demo), run 1

**Question:** How many loyalty points are required for a $25 discount, and when do points expire?

**Expected answer:** 500 points; points expire if the account is inactive for more than 12 months

**Reference passage (what the paper says):**
> 1 point for every dollar spent
> 100 points per $5 discount
> inactive for more than 12 months

**Model's answer:**

> According to the Acme Rewards program, 250 points are required for a $25 discount [1]. Points expire if an account is inactive for more than 12 months [1].

**Sources the answer cites:**

**[1] Acme Corp Customer Policy Handbook (file:company_policy) §Section 6: Loyalty Program, p.?**
> The Acme Rewards program allows customers to earn 1 point for every dollar spent. Points can be redeemed for discounts on future purchases at a rate of 100 points per $5 discount. Points expire if an account is inactive for more than 12 months. Loyalty program membership is free and does not require a minimum purchase.

verdict: partial
[1] supports: partly
note: expiry right; 250 points is wrong: 100 points per $5 means 500 for $25. [1] supports the expiry, not 250. An arithmetic slip with the right source in hand (generation failure, like Phase 12's loyalty case)

---
