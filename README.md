## PAPER LIST

The following is a list of recommended paper in many fields including 'sampling, generative modele, inference'.
Please email Zhihang Xu （xuzhh@shanghaitech.edu.cn） if you have any suggestions.

---
## Part I:  Sampling and generative models

### Topic 1: Markov Chain Monte Carlo sampling

#### Recommanded textbooks: 
- [Machine Learning: A probabilistic Perspective](https://doc.lagout.org/science/Artificial%20Intelligence/Machine%20learning/Machine%20Learning_%20A%20Probabilistic%20Perspective%20%5BMurphy%202012-08-24%5D.pdf)： Chapter 23 and 24.

#### Recommanded papers: 
- [Generalizing Hamiltonian Monte Carlo with Neural Networks](https://arxiv.org/abs/1711.09268),
 Daniel Levy, Matthew D. Hoffman and Jascha Sohl-Dickstein, 
 arXiv preprint arXiv:1711.09268, 2017.
 [[Code]](https://github.com/brain-research/l2hmc).

- [A-NICE-MC: Adversarial Training for MCMC](https://arxiv.org/abs/1706.07561),
Jiaming Song, Shengjia Zhao and Stefano Ermon,
31st Conference on Neural Information Processing Systems (NIPS 2017), Long Beach, CA, USA.
[[Website]](https://ermongroup.github.io/blog/a-nice-mc/).
[[Code]](https://github.com/ermongroup/a-nice-mc).


- [Learning Deep Latent Gaussian Models with Markov Chain Monte Carlo](https://pdfs.semanticscholar.org/353a/6ac63ba0f30f7627cb01e4ba214acf3a256c.pdf),
Matthew D. Hoffman,
Proceedings of the 34 th International Conference on Machine
Learning, Sydney, Australia, PMLR 70, 2017.

- [Auxiliary Variational MCMC](https://openreview.net/pdf?id=r1NJqsRctX),
Raza. Habib and David. Barber,
Published as a conference paper at ICLR 2019.
[[Code]](https://github.com/AVMCMC/AuxiliaryVariationalMCMC).

### Topic 2: Variational Inference

#### Recommanded textbooks: 
- [Pattern Recognation and Machine Learning](http://users.isr.ist.utl.pt/~wurmd/Livros/school/Bishop%20-%20Pattern%20Recognition%20And%20Machine%20Learning%20-%20Springer%20%202006.pdf): Chapter 10.1 and 10.2.
- [Machine Learning: A probabilistic Perspective](https://doc.lagout.org/science/Artificial%20Intelligence/Machine%20learning/Machine%20Learning_%20A%20Probabilistic%20Perspective%20%5BMurphy%202012-08-24%5D.pdf)： Chapter 21 and 22.

#### Recommanded papers: 
- [Markov Chain Monte Carlo and Variational Inference:
Bridging the Gap](http://proceedings.mlr.press/v37/salimans15.pdf),
Tim Salimans, Diederik P. Kingma and Max Welling,
Proceedings of the 32 nd International Conference on Machine
Learning, Lille, France, 2015.
[[Lecture slides]](http://videolectures.net/site/normal_dl/tag=1005141/icml2015_salimans_variational_inference_01.pdf)

- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114),
Diederik P Kingma and Max Welling,
The International Conference on Learning Representations (ICLR), Banff, 2014,
[[Video]](https://www.youtube.com/watch?v=rjZL7aguLAs),
[[Slides]](https://www.slideshare.net/mehdidc/auto-encodingvariationalbayes-54478304).

- [Hamiltonian Variational Auto-Encoder](https://arxiv.org/abs/1805.11328),
Anthony L. Caterini, Arnaud Doucet and Dino Sejdinovic,
32nd Conference on Neural Information Processing Systems (NIPS 2018),
[[Code]](https://github.com/anthonycaterini/hvae-nips),
[[Video]](https://www.youtube.com/watch?v=MD1CFKTu9U4).

- [Stein Variational Gradient Descent: A General
Purpose Bayesian Inference Algorithm](https://papers.nips.cc/paper/6338-stein-variational-gradient-descent-a-general-purpose-bayesian-inference-algorithm.pdf),
Qiang Liu and Dilin Wang,
30th Conference on Neural Information Processing Systems (NIPS 2016), Barcelona, Spain.
[[Code]](https://github.com/DartML/Stein-Variational-Gradient-Descent).



### Topic 3: Importance Sampling 

#### Recommanded papers: 
- [Coupling the reduced-order model and the generative model for an importance sampling estimator](https://arxiv.org/pdf/1901.07977.pdf)
Xiaoliang Wan and Shuangqing Wei,
arXiv preprint arXiv:1901.07977 (2019).

- [Neural Importance Sampling](https://arxiv.org/pdf/1808.03856.pdf),
THOMAS MÜLLER, BRIAN MCWILLIAMS, FABRICE ROUSSELLE, MARKUS GROSS, JAN NOVÁK.


### Topic 4: Generative Models 
Apart from VAE, GAN is an another typical methodology to generate samples from the “estimated(implicit)” distribution to estimate the target distribution if it works, there are some papers.

- [Generative Adversarial Nets](https://papers.nips.cc/paper/5423-generative-adversarial-nets)
Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville and Yoshua Bengio,
Advances in neural information processing systems(NIPS). 2014.
[[Code]](https://github.com/goodfeli/adversarial).

- [Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784)
Mehdi Mirza and Simon Osindero,
(arXiv preprint arXiv:1411.1784) 2014.

- [InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets](https://papers.nips.cc/paper/6399-infogan-interpretable-representation-learning-by-information-maximizing-generative-adversarial-nets.pdf),
Xi Chen, Yan Duan, Rein Houthooft, John Schulman, Ilya Sutskever and Pieter Abbeel,
30th Conference on Neural Information Processing Systems (NIPS 2016), Barcelona, Spain.

###  Topic 5:  Flow-based generative model
This topic focus on density estimation (or related) that is a central topic in unsupervised learning. Combining deep neural networks with the standard statistic method is very popular in recent years. The basic GAN model and its variants are omited here.

- [Density Estimation using real NVP](https://arxiv.org/pdf/1605.08803.pdf),
Laurent Dinh, Laurent Dinh and Laurent Dinh,
Published as a conference paper at ICLR 2017.

- [Flow-GAN: Combining Maximum Likelihood and Adversarial Learning in
Generative Models](https://arxiv.org/pdf/1705.08868.pdf),
Aditya Grover, Manik Dhar and Stefano Ermon,
Thirty-Second AAAI Conference on Artificial Intelligence. 2018.

- [Glow: Generative Flow with Invertible 1×1 Convolutions](https://papers.nips.cc/paper/8224-glow-generative-flow-with-invertible-1x1-convolutions.pdf),
Diederik P. Kingma and Prafulla Dhariwal,
Advances in Neural Information Processing Systems. 2018.

- [Neural Importance Sampling](https://arxiv.org/pdf/1808.03856.pdf),
THOMAS MÜLLER, BRIAN MCWILLIAMS, FABRICE ROUSSELLE, MARKUS GROSS, JAN NOVÁK.

- [Nice: non-linear independent components estimation](https://arxiv.org/pdf/1410.8516.pdf),
Laurent Dinh,David Krueger, Yoshua Bengio,
Accepted as a workshop contribution at ICLR 2015.

- [Monge-Ampère Flow for Generative Modeling](https://arxiv.org/abs/1809.10188),
Linfeng Zhang, Weinan E, Lei Wang,
[Code](https://github.com/wangleiphy/MongeAmpereFlow).

## Part II. Scientific Computing and Learning

### Topic 1: Integration
- [Efficient Monte Carlo Integration Using Boosted Decision Trees and Generative Deep Neural Networks](https://arxiv.org/pdf/1707.00028.pdf),
Joshua Bendavid,
Prepared for submission to JHEP.

### Topic 2: Optimization
- [Adam: A method for stochastic optimization](https://arxiv.org/pdf/1412.6980.pdf),
Diederik P. Kingma, Jimmy Lei Ba,
Published as a conference paper at ICLR 2015.

### Topic 3: Tensor and data-mining in scientific computing and data science
- [A practical randomized CP tensor decomposition](https://epubs.siam.org/doi/pdf/10.1137/17M1112303)
Battaglino C, Ballard G, Kolda T G.
SIAM Journal on Matrix Analysis and Applications, 2018, 39(2): 876-901.

- [Generalized CP decomposition](https://arxiv.org/pdf/1808.07452.pdf)
Hong D, Kolda T G, Duersch J A.
arXiv preprint arXiv:1808.07452, 2018.

- [Tensor Analyzers](http://www.cs.toronto.edu/~fritz/absps/ta.pdf)
Tang Y, Salakhutdinov R and Hinton G,
International Conference on Machine Learning (ICML). 2013.

- [Tensorizing neural networks](https://arxiv.org/abs/1509.06569)
Novikov A, Podoprikhin D, Osokin A, et al.
Advances in neural information processing systems (NIPS). 2015.

### Topic 4: Deep Learning based UQ analysis 

- [Uncertainty in neural networks: Bayesian Ensembling](https://arxiv.org/pdf/1810.05546.pdf),
Pearce T, Zaki M, Brintrup A, et al,
arXiv preprint arXiv:1810.05546, 2018.

- [Deep Neural Networks Motivated by Partial Differential Equations](https://arxiv.org/pdf/1804.04272.pdf)
Lars Ruthotto and Eldad Haber,
arXiv preprint arXiv:1804.04272, 2018.

- [Tensorizing neural networks](https://arxiv.org/abs/1509.06569)
Novikov A, Podoprikhin D, Osokin A, et al.
Advances in neural information processing systems (NIPS). 2015.

- [Learning Neural PDE Solvers with Convergence Guarantees](https://openreview.net/pdf?id=rklaWn0qK7)
Hsieh J T, Zhao S, Eismann S, et al.
[[Review Website]](https://openreview.net/forum?id=rklaWn0qK7).

### Topic 5: Coupling

- [Inference via Low-Dimensional Couplings](http://www.jmlr.org/papers/volume19/17-747/17-747.pdf),
Alessio Spantini, Daniele Bigoni and Youssef Marzouk,
Journal of Machine Learning Research 19 (2018) 1-71.

### Topic 6: ANOVA
- [Calibration of Computational Models with Categorical Parameters and Correlated Outputs via Bayesian Smoothing Spline ANOVA](https://arxiv.org/abs/1405.5297),
Curtis B. Storlie, William A. Lane, Emily M. Ryan, James R. Gattiker, David M. Higdon.

---
### Some Reading List of relevent fields
- [Variational Inference](http://www.statslab.cam.ac.uk/~sp825/vi.html)
- [Bayesian Deep Learning & Deep Bayesian Learning](https://github.com/CW-Huang/BDL-Reading-List/blob/master/index.md)
- [Applying Machine Learning To Physics](https://physicsml.github.io/pages/papers.html)


