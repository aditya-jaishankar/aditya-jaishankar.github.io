---
title: "Fundamentals of Statistics(18.6501x)"
categories:
toc: true
layout: single
classes: wide
permalink: /coursenotes/statistics/
author_profile: true
toc: true
date: February 2019
read_time: true
---

## Unit 1: Introduction to Statistics

### Lecture 1: What is Statistics

* Statistics is at the core, is fundamental to machine learning and data science. Without a thorough grounding in statistics, machine learning becomes a black box. 

* There is the **computational view** of data: where problems are solved with very large amounts of data, and techniques such as spectral methods, low dimensional embedding (visualizing high dimensional data), distributed optimization, etc. Then there is the **statistical view** of data, where data is treated as a random variable, and each data point is the realization of a random variable, i.e., they are outcomes of a random process. Statistics essentially allows you to answer the question of, given this small amount of data, what accurate statements can I make about the case if I had an infinite amount of data.  

* Probability and Statistics go hand in hand. Given the truth, probability lets us calculate what kind of data we should expect. Statistics on the other hand is the reverse - given a set of observations (data) what can we say about the truth, and with what level of certainty can we say that? Statistics is the process of trying to determine the truth from a set of observations. 

### Lecture 2: Probability redux

**Let us recall some basic inequalities, in order of conservatism**

In all of this, we assume that $\bar{X} _n$ is the sample mean, each of the $X_i$ are i.i.d r.v's. 

* Markov's inequality: 

    Given a non negative random varialbe $X$, the inequality states that

    $$
    \begin{align}
        \mathbf{P}(X \geq a) \leq  \dfrac{\mathbf{E}[X]}{a}
    \end{align}
    $$

* Chebychev's inequality: 

    If $X$ is a random variable with *finite expected value* $\mu$ and non-zero variance $\sigma^2$ (can be unbounded), then for some $\varepsilon > 0$, 

    $$
    \begin{align}
        \mathbf{P}(\lvert X - \mu \rvert > \varepsilon) \leq \dfrac{\sigma^2}{\varepsilon^2}
    \end{align}
    $$

    Note that the weak law of large numbers (that the sample mean converges in probability to the mean of the distribution) follows easily from the application of Chebychev's inequality. 

* Hoeffding's inequality:

    Let $X_1, X_2, \ldots, X_n$ be independent identically distributed random variables that are almost surely bounded on the interval $[a, b]$. Further, let $\bar{X} _n$ be the sample mean. Then, the inequality states that:

    $$
    \begin{align}
        \mathbf{P}(\bar{X} _n - \mathbf{E}[X] \geq \epsilon) \leq 2\exp\left(-\dfrac{2 n \epsilon^2}{(b - a)^2}\right)
    \end{align}
    $$ 

    for all $\epsilon > 0$. The sample size $n$ does not need to be large. 

* Central limit theorem:

    Let $X_1, X_2, \ldots, X_n$ be independent identically distributed random variables, with *finite* mean $\mu$ and *finite* variance $\sigma^2$. The CLT states that

    $$
    \begin{align}
    \lim _{n \to \infty} \mathbf{P}\left(\dfrac{\sum X_i -n\mu}{\sqrt{n}\sigma} \leq z \right) = \Phi(z)
    \end{align}
    $$

    where $\Phi(\cdot)$ denotes the CDF of the standard normal. 

* The conservatism decreases in the order shown because as we go down the list, we start to know more and more about the random variables (markov - only the mean, Chebychev - mean and variance, Hoeffding - mean and boundedness). In the case of the CLT, we have an infinite number of random variables which gives us the tightest bounds.

* Affine transformation of the Gaussian distribution: If $X \sim N(\mu, \sigma^2)$, then $aX + b \sim N(a\mu + b, a^2\sigma^2)$. 

* Standardization of the Gaussian distribution: If $X \sim N(\mu, \sigma^2)$, then $Z = (X - \mu)/\sigma$ (sometimes called the $Z$ score) is a standard normal. 

* Symmetry of the Gaussian distribution: If $X \sim N(0, \sigma^2)$ then $-X \sim N(0, \sigma^2)$. The probability distributions are the same. 

* The quantile of a random variable $X$ of order $1-\alpha$ is the number $q_\alpha$ such that $\mathbf{P}(X \geq q_\alpha) = \alpha$ i.e., $\mathbf{P}(X \leq q_\alpha) = 1 - \alpha$. 

* Can you recall convergence almost surely, convergence in probability, and convergence in distribution. Convergence a.s. is the strongest, then convergence in probability theorem, then convergence in distribution. Can you explain these intuitively? See [this](https://stats.stackexchange.com/questions/2230/convergence-in-probability-vs-almost-sure-convergence) discussion for a really nice intutive discussion if you cannot recall.

* If $T_n \xrightarrow[n \to \infty]{\textrm{a.s}/ \mathbf{P}} T$ and $U_n \xrightarrow[n \to \infty]{\textrm{a.s}, \mathbf{P}} U$, then things are as you would expect and

    $$
    \begin{align}
        T_n \cdot U_n & \xrightarrow[n \to \infty]{\textrm{a.s}/ \mathbf{P}} T U\\
        T_n + U_n & \xrightarrow[n \to \infty]{\textrm{a.s}/ \mathbf{P}} T + U \\
        T_n / U_n & \xrightarrow[n \to \infty]{\textrm{a.s}/ \mathbf{P}} T/U, U \neq 0
    \end{align}
    $$

* The above is in general not true for convergence in distribution! However, **Slutsky's theorem** states that If $T_n \xrightarrow[n \to \infty]{(d)} T$ and $U_n \xrightarrow[n \to \infty]{\mathbf{P}} u$, where $u$ is a constant number, then the above results are true (with the replacement $U \to u$). 

* **Continuous mapping theorem:** If $T_n \xrightarrow[n \to \infty]{\textrm{a.s}/ \mathbf{P}/(d)} T$, then $f(T_n) \xrightarrow[n \to \infty]{\textrm{a.s}/ \mathbf{P}/(d)} f(T)$, if the function $f$ is continuous. 

## Unit 2: Parametric Inference

### Parametric Statistical Models

* The **holy trinity of statistical inference** is 1. Estimation, 2. Confidence intervals, and 3. Hypothesis Testing.

* **Definition - Statistical Model**: A statistical model is defined in terms of a pair 
    
    $$
    \begin{align}
    (E, \{\mathbb{P} _\theta\}_{\theta \in \Theta}) \label{eqn:modeldefn}
    \end{align}
    $$

    where $E$ is a measureable space known as the sample space (i.e. the set of values the random variable in question can take), $\\{\mathbb{P}_\theta\\}$ is a family of probability distributions with parameters $\theta$. The domain (space) of possible values that $\theta$ can take is denoted by $\Theta$. For example, for a Bernoulli distribution, $E = \\{0, 1\\}$, $\theta$ is the parameter of the distribution, and $\Theta$ is the interval $[0, 1]$.

* Here we will only study models that are well-specified, i.e., if $\mathbb{P}$ is the true distribution that generated the data, then if the model is well-specified, there exists $\theta \in \Theta$ such that $\mathbb{P}_\theta = \mathbb{P}$. The value of $\theta$ is unknown (also sometimes called the true parameter) and the aim of the statistical experiment is to estimate $\theta$.

* $\Theta \subseteq \mathbb{R}^d, d \geq 1$ in which case the model is called parametric. Sometimes, $\Theta$ can be infinite dimensional, in which case the model is called non-parametric.  In other words, a statistical model is called parametric if all parameters $\theta \in \Theta$ is specified by a finite number of unknowns. In particular, $\Theta \subset \mathbb{R}^m$, and the family of probability distributions $\\{ \mathbb{P}_\theta \\}$ is specified by the $m$ components of the vector $\theta$. 

* To summarize: A statistical model consists of a sample space $E$ and a family of probability models $\\{\mathbb{P}_\theta\\}$ indexed by the parameter $\theta$, where $\theta \in \Theta$. The notation for denoting a statistical model is given in equation \ref{eqn:modeldefn}. *Note that the sample space cannot contain any unknown parameters; for example in a uniform distribution $U(0,a)$ with $a>0$ unknown, I cannot say that the sample space is $(0, a)$ because $a$ is unknown.* 

* **Examples of statistical model notation:**

    Bernoulli Trials: $(\\{0, 1\\}, \\{\textrm{Ber}(p)\\} _{p \in (0, 1)})$. 

    Poisson Model: $(\mathbb{N}^0, \\{\textrm{Poiss}(\lambda)\\} _{\lambda \in \mathbb{R}^+})$

    If $X_1, \ldots, X_n$ (i.i.d) $\tilde N(\mu, \sigma^2)$, for $\mu \in \mathbb{R}$ and $\sigma^2 > 0$., $(\mathbb{R}, \\{N(\mu, \sigma^2)\\} _{(\mu, \sigma^2) \in \mathbb{R} \times (0, \infty)})$.

    There are also more complicated models for example Linear Regression. In this model, we have $(X_1, Y_1), ldots,(X_n, Y_n) \in \mathbb{R}^d \times \mathbb{R}$, with $Y_i = \beta^T X_i + \epsilon_i$, $\epsilon_i \sim N(0, 1)$ and $\beta \subset \mathbb{R}^d$, $X_i \sim N(0, I_d)$ where $I_d$ is the identity vector. You see that the sample space and $\Theta$ are easy to write out in this case, but $\mathbb{P} \_\theta$ is significantly harder. We will need to first find $f_{Y_i \lvert X_i}(y_i \lvert x_i)$ and then determine $f_{X_i}(x_i)$ through standard techniques learned in the probability class (start with the CDF and then differentiate).

* Identifiability: A statistical model is said to be identifiable if $\theta \neq \theta' \Rightarrow \mathbb{P}\_\theta \neq \mathbb{P}\_\theta'$. In other words, given the probability distribution,  I can uniquely identify the parameter set. 

* In general, when trying to understand identifiability of a parameter in a parametric model, try and calculate the CDF or the PDF of and see if the pdf is an injective function i.e. if $f(x) = f(y) \Rightarrow x = y$. 

### Parametric estimation and confidence intervals

* **Definitions:**

    * Statistic: Any measureable function of the samples, for example $\bar{X}_n$ or $X_1 + \log (1+X_n^2)$ and so on. *Measureable* just says that given data, I should be able to compute the value

    * Estimator: Any statistic that does not depend on the parameter $\theta$. The goal is designing an estimator is that I want to find a distribution that is as close as possible to the true distribution. I want to ensure that $\hat{\theta}_n$ converges to $\theta$ given enough samples. 

    * Weak consistency: An estimator $\hat{\theta}_n$ of $\theta$ is said to be weakly consistent if $\theta_n \xrightarrow[n \to \infty]{\mathbb{P}} \theta$. 

    * Strong consistency: An estimator $\hat{\theta}$ of $\theta$ is said to be strongly consistent if $\theta_n \xrightarrow[n \to \infty]{\textrm{a.s.}} \theta$.

    * Asymptotic normality: An estimator $\hat{\theta}_n$ of $\theta$ is said to be asymptotically normal if 

    $$
    \begin{align}
        \sqrt{n}(\hat{\theta}_n - \theta) \xrightarrow[n \to \infty]{(d)} N(0,\sigma^2)
    \end{align}
    $$

    Here, $\sigma^2$ is called its asymptotic variance.

* Bias of an estimator: The bias of an esitmator $\hat{\theta}_n$ of $\theta$, denoted by $\textrm{bias}(\hat{\theta}_n)$ is defined by $\mathbb{E}[\hat{\theta}_n] - \theta$. If the bias is 0, the estimator is called unbiased. Having an unbiased estimator is desirable, but should not be pursued at all consts. One could ontain an unbiased estimator but end up habving a very large variance for the estimator, so we it's estimates might be fraught with uncertainty. Therefore, while judging how good estimators are, we also need to compute the variance of estimators.

* Jensen's inequality: If $g(\cdot)$ is a convex function, then $g(\mathbf{E}[X]) \leq \mathbf{E}[g(X)]$. For a concave function, this inequality flips in direction.

* Quadratic risk: This is a way to try and optimize both the bias and the variance of the estimator. It is defined as $\mathbb{E}[\lvert \hat{\theta}_n - \theta \rvert^2]$. It is easy to derive that this quantity is equal to $\textrm{var}(\hat{\theta}_n - \theta) + (\mathbb{E}[\hat{\theta}_n] - \theta)^2$ which is the sum of the variance and the square of the bias. Here is an interesting insight: if the quadratic risk of $\hat{\theta}_n \xrightarrow[n \to \infty]{ } 0$, then the sum of the variance and the square of the bias goes to 0. If the sum of two non-negative quantites goes to zero, each must go to zero. This implies that th variance tends to 0 and the expected value of $\hat{\theta}_n$ tends to $\theta$. So the peak in the distribution becomes narrower and narrower, the location of the peak approaches $\theta$. This means that the estimator converges in probability to the parameter $\theta$. 

* Note: There are lots of potential advantages of using the sample mean as an estimator:
    * *It is consistent*, that is, $\hat{\theta}_n = \bar{X}_n \xrightarrow[n \to \infty] \theta$. Using Chebychev, it is easy show that $\bar{X}_n$ converges in probability to $\theta$. It is also trivial to show that the variance goes to 0 with increasing $n$ (assuming iid). So we have a distribution centered at $\theta$ with zero variance, so it is $\theta$ itself. 

    * *It is unbiased*: Use linearity of expectations.

    * It is *efficiently computable*: Given the samples $X_i$ it is very easy to calculate.

    * The quadratic risk goes to 0 as $n \to \infty$. Both the variance and the bias go to 0 as $n \to \infty$ so this statement trivially follows. 

* Calculating confidence intervals can be tricky because the requirement is that the confidence interval should not contain any parameters. The confidence interval cannot be a function of the parameters. If we were to simply use the central limit theorem, we include the standard deviation of the probabilistic model, and for example if we were using a Bernoulli model, then $p$ appears in th calculation for the confidence interval. i.e. I would want to solve for $x$ such that

    $$
    \begin{align}
        \mathbb{P}(\lvert \bar{R}_n - p \rvert \geq x) = 2\left(1 - \Phi\left(\dfrac{\sqrt{n}x}{\sqrt{p(1-p)}}\right)\right) = \alpha
    \end{align}
    $$
    
    and $p$ now appears. So how do we get around this?

* To get around this, we need to remind ourselves of quantiles, set up the problem and the notation and then proceed to see what we are going to do. Recall that the $q_{\alpha/2}$ is the $1-(\alpha/2)$ quantile of $X \sim N(0,1)$ if $\mathbb{P}(X > q_{\alpha/2}) = \alpha/2$. Now in this case we are interested in the random variable $X = \bar{R_n} - p$, i.e.,  how far our estimator is from our the estimate we are trying to determine. Therefore, 

    $$
    \begin{align}
        \mathbb{P}(\lvert \bar{R}_n - p \rvert \leq q_{\alpha/2}) & = 1- \alpha
    \end{align}
    $$

    Note that the factor of 3 disappears on the right hand side when we remove the aboslute value symbol. The above equation after some algebra can be rearranged to

    $$
    \begin{align}
        \mathbb{P} \left(\lvert \bar{R}_n - p \rvert \leq \dfrac{\sqrt{p(1-p)}q _{\alpha/2}}{\sqrt{n}}\right) = 1 - \alpha
    \end{align}
    $$

    In other words, with probability  $\approx 1 - \alpha$,

    $$
    \begin{align}
        \bar{R}_n \in \left[p - \dfrac{\sqrt{p(1-p)}q _{\alpha/2}}{\sqrt{n}}, p + \dfrac{\sqrt{p(1-p)}q _{\alpha/2}}{\sqrt{n}} \right]
    \end{align}
    $$ 

* The first technique to eliminate $p$ is the **conservative bound**. We set $p$ to be its value at maximum variance. For the Bernoulli, $p=1/2$, and hence $\sqrt{p(1-p)} = 1/2$. Then we can say that with probability at least $1-\alpha$, 

    $$
    \begin{align}
        \bar{R}_n \in \left[p - \dfrac{q _{\alpha/2}}{2\sqrt{n}}, p + \dfrac{q _{\alpha/2}}{2\sqrt{n}} \right]
    \end{align}
    $$ 

    We therefore get the asymptotic confidence interval

    $$
    \begin{align}
        \mathcal{I}_\textrm{conserv} = \left[\bar{R} _n - \dfrac{q _{\alpha/2}}{2\sqrt{n}}, \bar{R} _n + \dfrac{q_{\alpha/2}}{2\sqrt{n}} \right]
    \end{align}
    $$

    More precisely, 

    $$
    \begin{align}
        \lim_{n\to \infty} \mathbb{P}(\mathcal{I}_\textrm{conserv} \ni p) \geq 1 - \alpha
    \end{align}
    $$

* The second method is to actually solve for $p$ using a quadratic equation (for Bernoulli) or more complicated equations for other models. For Bernoulli, we get a new interval $\mathcal{I}_\textrm{solve} = [p_1, P_2]$, where $p_1, p_2$ are roots of the quadratic. More precisely, with this solution, 

    $$
    \begin{align}
        \lim_{n \to \infty} \mathbb{P}(\mathcal{I}_\textrm{solve} \ni p) = 1 - \alpha
    \end{align}
    $$

    Note that this is a strong equality. There are no approximations. We just require the central limit theorem to hold, which of course it does. 

* The third method to get at a confidence interval that does not contain the parameter $p$ that we are trying to estimate is to to do a plug in. This proceeds as follows. Consider the quantity that I am interested in calculating

    $$
    \begin{align}
        \dfrac{\sqrt{n}(\bar{R}_n - p)}{\sqrt{p(1-p)}}
    \end{align}
    $$

    Now form the law of large numbers, we know that $\bar{R}_n$ converges to $p$, i.e. $\hat{p} \xrightarrow[n \to \infty]{\mathbb{P}, \textrm{a.s.}} p$. Therefore, using the continuous mapping theorem, 

    $$
    \begin{align}
        \dfrac{\sqrt{\hat{p}(1-\hat{p})}}{\sqrt{p(1-p)}} \xrightarrow[n \to \infty] 1
    \end{align}
    $$

    Now, since this converges in proability (and almost surely), we can use Slutsky's theorem:

    $$
    \begin{align}
        \dfrac{\sqrt{n}(\bar{R}_n - p)}{\sqrt{p(1-p)}} & = 
        \dfrac{\sqrt{n}(\bar{R}_n - p)}{\sqrt{\hat{p}(1-\hat{p})}}\cdot\dfrac{\sqrt{\hat{p}(1-\hat{p})}}{\sqrt{p(1-p)}} \\
        & \xrightarrow[n \to \infty]{(d)} N(0,1)
    \end{align}
    $$

    So we get a new confidence interval $\mathcal{I}_\textrm{plug-in}$ given by

    $$
    \begin{align}
        \mathcal{I}_\textrm{plug-in} = \left[\bar{R}_n - \dfrac{\sqrt{\hat{p}(1-\hat{p})}q _{\alpha/2}}{\sqrt{n}}, \bar{R}_n + \dfrac{\sqrt{\hat{p}(1-\hat{p})}q _{\alpha/2}}{\sqrt{n}} \right]
    \end{align}
    $$

    Such that

    $$
    \begin{align}
        \lim_{n \to \infty} \mathbb{P}(\mathcal{I}_\textrm{plug-in} \ni p) = 1 - \alpha
    \end{align}
    $$

    A general note: *When replacing out things that you don't know with things that you know by putting a hat on it, you need to think of slutsky and limits and the continuous mapping theorem. This is especially useful when calculating confidence intervals using* $\mathcal{I}_\textrm{plug-in}$. 

* Summary: Given some unknown but fixed parameter $\theta \in \mathbb{R}$ for a parametric model and $X_1, \ldots, X_n$ distributed I.i.d. $\mathbb{P}_\theta$, the non-asymptotic 95% confidence interval of $p$ is an interval $\mathcal{I} = \mathcal{I}(X_1, \ldots, X_n)$ such that $\mathbf{P}(\mathcal{I} \ni \theta) \geq 95$%. It is important to note that $\mathcal{I}$ is random, because it depends on some function of random variables. One the other hand, the realization of a random variable is deterministic. Therefore, say I am given the interval $[0.37, 0.54]$ as a 95% confidence interval for $p$ and then asked for the probability that $p$ is in this interval, *it is not 95%*. In fact, this is a deterministic question. Either $p$ falls in this interval and the probability is 1, or $p$ falls outside this interval, and the probability is 0. So, when determining confidence intervals, we use a process to say that I constructed a random variable, and now I make a statement about the outcome of conducting an experiment on that random variable. Once the experiment has been carried out, there are no more probabilities involved. I just check to see if the high probability event happened, or the low probability event happened. Unlikely events happen all the time, so it is possible (but not probable) that the true value of $p$ does not lie in the 95% confidence interval. Once I get numbers for my estimator $\hat{\theta}$, the question becomes deterministic.

* If asking whether $\theta \in [\hat{\theta}_1, \hat{\theta}_2]$ is a deterministic question, what does a 95% confidence interval really mean? The frequentist view of the confidence interval says that if I do an experiment over and over and over again i.e. I determine a whole bunch of confidence inervals $\mathcal{I}$, and further, if I define a Bernoulli random variable which takes value one if it turns out that $\mathcal{I} \ni \theta$, then the parameter of the Bernoulli random variable would be 0.95. In other words, I would find that 95% of the time, $\mathcal{I} \ni \theta$. 

### Lecture 5: Delta method and confidence intervals

* **The Delta method**

    The delta method allows us to apply functions to an estimator we construct so that we can estimate values of interest rather functions of the quantity being estimated. We will discuss an example next, but first let us look at the formal definition. If 

    $$
    \begin{align}
        \sqrt{n}(\bar{R}_n - \theta) \xrightarrow[n \to \infty]{(d)} N(0, \sigma^2)
    \end{align}
    $$

    then the delta method states that

    $$
    \begin{align}
        \sqrt{n}(g(\bar{R}_n) - g(\theta)) \xrightarrow[n \to \infty]{(d)} N(0,(g'(\theta))^2 \sigma^2)
    \end{align}
    $$

    provided that $g'(\theta)$ is continuously differentiable, that is $g'(\theta)$ exists and is continuous. Let us not look at an example.

* Consider $X_1, \dots, X_n$ to be i.i.d. $\exp(\lambda)$. How do we construct an estimator for the parameter $\lambda$? We fist need to check for consistency and then to see if it unbiased. The first insight is to start with what we know quite well: the sample mean. We know from the weak law of large numbers that $\bar{X}_n \xrightarrow[n \to \infty]{\mathbf{P}} \mathbf{E}[X_i] = 1 / \lambda$. We can now use the continuous mapping theorem using the function $f(x) = x^{-1}$ to say that

    $$
        \dfrac{1}{\bar{X}_n} \xrightarrow[n \to \infty]{\mathbf{P}} \lambda. 
    $$

    Therefore, the estimator $\hat{\lambda} = 1/\bar{X}_n$ is consistent. Is it unbiased? Is $1/\bar{X}_n - \lambda = 0$? This can be calculated and it turns out that this is not an unbiased estimator. A simple way to see this would be to use Jensen's inequality. $g(x) = 1/x$ is a convex function and hence $g(\mathbf{E}[x]) < \mathbf{E}[g(x)]$. (The equality only holds if the function is affine (i.e. linear transformation + constant). An unbiased estimator is a desirable property but should not be pursued at all costs (for example at the expense of a very large variance.)

    Now, we know from the central limit theorem that

    $$
        \sqrt{n}(\bar{X}_n - 1/\lambda) \xrightarrow[n \to \infty]{\mathbf{P}} N(0, 1/\lambda^2)
    $$

    We can now apply the delta method. Using $g(x) = 1/x$, $g'(1/\lambda) = - \lambda^2$. So applying the delta method yields

    $$
        \sqrt{n}(1/\bar{X}_n - \lambda) \xrightarrow[n \to \infty]{\mathbf{P}} N(0, \lambda^4/\lambda^2)
    $$

    Now, let's say that we need to find confidence intervals for this estimator? Calling $1/\bar{X}_n = \hat{\lambda}$, we have from the above equation that

     $$
        \sqrt{n}(\hat{\lambda} - \lambda) \xrightarrow[n \to \infty]{\mathbf{P}} N(0, \lambda^4/\lambda^2)
    $$

    and from the central limit theorem, we have that the confidence interval at level alpha is defined by 

    $$
        \dfrac{\sqrt{n}\lvert \hat{\lambda} - \lambda \rvert }{\lambda} \leq q_{\alpha/2}
    $$

    and now we can either use $I_\textrm{plug-in}$ or $I_\textrm{solve}$ to find the confidence interval. Exercise, can you now work out the case where the estimator is $\hat{\lambda} = \min(X_1, \ldots, X_n)$. Again, the steps are to show that it is consistent, find it's asymptotic variance and then calculate the confidence interval using with the conservative bound, plug-in, or solve technique.  



### Lecture 6: Introduction to hypothesis testing, and type 1 and type 2 errors

There is a lot of new terminology and new definitions in this unit here, so pay attention! The goal of this lecture is to introduce hypothesis testing. When doing hypothesis testing we are not interested in obtaining an estimator unknown parameters. We are only interested in asking binary questions about unknown parameters. Is the mean of average heights in the US larger or smaller than 5.5, is the waiting time in the ER less than 30 mins, etc.

* The example we are going to see here is to test between different boarding methods for flights. Let's say we are interested in evaluating two different methods - R2F (rear to front) or WilMA (window, middle, aisle in that order, which is an inside to outside boarding method). Let's, let's say we are given the following data (usually called summary statistics)

    |  | R2F | WilMA |
    |-------|--------:|---------:|
    | Mean (mins) | 24.2 | 15.9 |
    | Std. Dev. (mins) | 2.1 | 1.3 |
    | Samples | 72 | 56 |

    We now ask the question about whether the difference between means is significant? How do we quantify and evaluate this statistical significance, if any? 

* What are some of the modeling assumptions we need to begin to answer this question? 

    0. Let $X$ and $Y$ denote the boarding time of a random R2F and WilMA flight respectively. Further, we assume that $X \sim N(\mu_1, \sigma_1^2)$ and $Y \sim N(\mu_2, \sigma_2^2)$. Let $n$ and $m$ denote the sample sizes of $X$ and $Y$ respectively. We assume that $X_1, \dots, X_n$ and $Y_1, \dots, Y_n$  are independent **copies** (meaning identically distributed) of $X$ and $Y$ respectively. Further the two samples themselves are independent. In effect, we assume that every random variable is independent from every other random variable in the problem.

    0. We now ask the question whether $\mu_1 = \mu_2$ or $\mu_1 > \mu_2$. Note that with modeling assumptions, we decrease the number of ways the hypothesis $\mu_1 = \mu_2$ may be rejected. We do not allow for the case $\mu_1 < \mu_2$.

    0. We have two samples, so this is a **two sample test**. 

* Note that although I have data, I cannot make an assumption about which hypothesis is better just by looking at the data. I have to still assume that $\mu_1 = \mu_2$ is a possibility. If I make assumptions after looking at the data, I will make biased assumptions and of course my assumptions will play out. 

* So how do we actually go about this? As a starting heuristic, I could simply say that I can going to compare $\bar{X}_n$ and $\bar{Y}_m$, and if $\bar{X}_n > \bar{Y}_m $, then I could say that this is enough to conclude that $\mu_1 > \mu_2$. But this has some issues: what if $\mu_1 = 50.1$ and $\mu_1 = 50$. Or what if we had a sample size of 2? Then intuitively this heuristic doesn't make much sense.  We somehow need to capture a sense for the variability of the data and the how big our sample size is. So we are going to come up with a heuristic that says if

    $$
        \bar{X}_n - \textrm{buffer}_n > \bar{Y}_m + \textrm{buffer}_m
    $$

    then $\mu_1 > \mu_2$. The idea here is to capture the fact that there is some fluctuation of $\bar{X}_n$ and $\bar{Y}_m$ about their respective sample means $\mu_1$ and $\mu_2$. Note that the sizes of these buffers must go to 0 as $n, m \to \infty$ because if we have an infinite amount of data, then we know everything there is to know about the random variables $X$ and $Y$, and there is no uncertainty or randomness. 

* The idea here is best seen with an example. Let's say we toss a coin 200 times and we find that we get 80 heads. We now ask the question "is the coin fair?" More formally we are asking the question "Is $p = 0.5$?". To go about we look at what would happen if in fact $p = 0.5$ so we will go ahead and assume this. Let us also assume a sample mean $\bar{X}_n$ and be definition, this equals 80/200.  Then, because each coin toss is independent, assuming $p=0.5$ amounts to saying that some normal random variable took on the value defined by 

    $$
    \begin{align}
        \dfrac{\sqrt{n}(\bar{X}_n - 0.5)}{\sqrt{0.5(1-0.5)}}  & = \dfrac{\sqrt{n}(80/200 - 0.5)}{\sqrt{0.5(1-0.5)}} = -2.8284
    \end{align}
    $$

    In other words, we are syaing that if $p = 0.5$, then observing a sample mean of 80/200 is the same as oberving some standard normal random variable take on the value -2.8284. This turns out to be extremely unlikely, because this number is way out in the tail of the distribution. So the appropritate conclusion of this experiment is *it is unlikely that the coin is fair*. If you were to repeat the calculation and say that we say 106 heads in the 200 tosses, then this amounts to a standard normal taking on the value 0.8485, which is not that far in the tail of the distribution. In this case, we would conclude *it is likely that the coin is fair*. 

* We now introduce the concepts of **null hypothesis** $H_0$ and **alternate hypothesis** $H_1$ formally defined by

    $$
    \begin{cases}
        H_0: \theta \in \Theta_0 \\
        H_1: \theta \in \Theta_1
    \end{cases}
    $$

    The standard practice is to chose the null hypothesis as the status quo. For example, when checking to see if a new drug is any good, we choose as the null hypothesis that the drug is just as good as the placebo i.e. if I model the outcome of asking the question "Do you feel better" to the patients at the end of the trial as a $\textrm{Ber}(p)$, then the null hypothesis is $p_\textrm{drug} = p_\textrm{control}$. In this example, and always, we only look for evidence in the data that can falsify our null hypothesis. We look for falsifiability, we don't look for proof of our hypothesis in our data. In other words, all we can say is "we did not find evidence that H_0" is false. This is not the same sa saying that $H_0$ is true. Keep the innocent until proven guilty idea in mind. A jury can concludes, "not guilty" meaning, there was not enough evidence to show that he was guilty. They do not conclude "innocent". 

* One more definition: A *test* is a statistic $\psi$ such that 

    $$
    \psi = 
    \begin{cases}
        0, \textrm{ if } H_0 \textrm{ is not rejected} \\ 
        1, \textrm{ if } H_0 \textrm{ is rejected}
    \end{cases}
    $$

    In oher words, if $\psi = 1$, we reject the null hypothesis and if $\psi = 0$, we will fail to reject the null hypothesis (doesn't mean that the null hypothesis is correct). We can now formulate the test by comparing to a standard normal by saying that (using an example of a Bernoulli)

    $$
        \psi = \mathcal{I} \left\{\dfrac{\sqrt{n} \lvert \bar{X}_n - p \rvert}{\sqrt{p(1-p)}} > C \right\}
    $$

    for some $C$. he question now is how do we choose $C$? Note that we can always write a test $\psi$ as an indicator function. We are asking a binary question. Was the null hypothesis rejected? Yes or no? At the end of the day, it is also a function that can be computed from data. 

* We now also introduce **Type 1 errors** and **Type 2 errors**.
    * Type 1 errors are the case when my test rejects $H_0$ when it is actually true.

    * Type 2 errors are the case when my test fails to reject $H_0$ when $H_1$ is actually true. 

* Some more formalism: The rejection region of a test $\psi$ is defined by 

    $$
        R_\psi = \{x \in E^n: \psi(x) = 1\}
    $$

    Here the $E^n$ just means that $x$ is a vector of dimension $n$. It is just where $x_1, \ldots, x_n$ lives.

* Here are a whole bunch of definitions, which are extremely important. 

    * $\alpha_\psi$ is the **Type 1 error** which is a function of $\theta$:

        $$
            \begin{align}
                \alpha_\psi: \Theta_0 & \xrightarrow[ ]{ } \mathbb{R} \\
                \theta & \longmapsto \mathbf{P}_\theta(\psi = 1)
            \end{align}
        $$

        The meaning of the notation is that this function of $\theta$ maps every $\theta \in \Theta_0$ to a probability that lies in the interval (0,1). It calculates the probability that $\psi = 1$ when $\theta \in \Theta_0$. In other words, it calculates as a function of $\theta$ the probability that the test $\psi$ rejects $H_0$ when $H_0$ is in fact true. 
    
    * $\beta_\psi$ is the **Type 2 error** which is also a function of $\theta$:

        $$
            \begin{align}
                \beta_\psi: \Theta_1 & \xrightarrow[ ]{ } \mathbb{R} \\
                \theta & \longmapsto \mathbf{P}_\theta(\psi = 0)
            \end{align}
        $$

        The meaning here is: this function maps every $\theta \in \Theta_1$ to a probability that lies in the interval (0,1). It calculates the probaility that $\psi = 0$ when $\theta \in \Theta_1$. In other words, it calculates as a function o $\theta$ the probability that the test $\psi$ fails to reject $H_0$ when $H_1$ is infact true. 

    * Power of a test $\pi_\psi$:

    $$
    \begin{align}
        \pi_\psi = \inf\limits_{\theta \in \Theta_1} (1 - \beta_\psi(\theta))
    \end{align}
    $$

    * A test $\psi$ has level $\alpha$ if

    $$
        \alpha_\psi \leq \alpha \ \ \ \forall \theta \in \Theta_0
    $$

    * A test $\psi$ has asymptotic level $\alpha$ if

    $$
        \lim\limits_{n \to \infty} \alpha_{\psi_n} \leq \alpha \ \ \ \forall \theta \in \Theta_0
    $$

    * In general a test has the form 

    $$
        \psi = \mathbb{I}(T_n > C)
    $$

    where $\mathbb{I}$ is the indicator function, which equals one if the condition $T_n > C$ is satisfied. The quantity $T_n$ is called a test statistic, and the number $C$ is a threshold yet to be determined. The rejection region $R_\psi = \{T_n > C\}$. 

* In the case when $H_0$ is of the form $\theta \in \Theta_0$, $\Theta_0 = \{\theta_0\}$, we can refine the terminalogy further and define *one-sided* and *two-sided* tests. If $H_1: \theta \neq \theta_0$, then it is called a two-sided test. If $H_1: \theta > \theta_0$ or $H_1: \theta <> \theta_0$, then it is called a two-sided test. This terminology is useful when determining type 1 and type 2 errors because it tells us if we need to use absolute values in our standardization, and hence whether there is a factor of 2 floating around. 

* Note that when we are deciding on the threshold $C$

## Unit 3: Methods of estimation

### Lecture 8: Distance measures between distributions

* Maximum likelihood estimatiion should be the go to method when presented with some new problem at hand. Using the method of moments (say expectation or variance) often works, but what if I have multiple parameters, and if the expectation is a function of the multiple parameters. In this case, maximum likelihood estimation should be the standard method I should reach for. In the special cases where the method of moments works, I should still attempt that becuase of its simplicity. 

* Because I am calculating maxima, maximum likelihood estimation can sometimes be computationally intractable. 

* The goal of a statistician is to be able to estimate a probability distribution. Let's say that there is some true probability distribution $$ \mathbb{P}_ {\theta^*} $$ that is generating the data, and that I have the model $$ \mathbb{P}_ \theta $$. I want $$ \mathbb{P}_ \theta $$ to be as close as possible to $$ \mathbb{P}_ {\theta^*} $$ for all possible $x \subset E$. This naturally leads to the notion of a *total variation distance* which computes

    $$
    \begin{align}
        TV(P_\theta, P_{\theta'}) = \max\limits_{A \subset E}\lvert P_\theta - P_{\theta'} \rvert
    \end{align}
    $$
    
    It turns out for a discrete probability distribution, there is a nice analytical form for this and the total variation distance becomes

    $$
    \begin{align}
        TV(P_\theta, P_{\theta'}) = \dfrac{1}{2}\sum\limits_{x \in E} \lvert p_\theta (x) - p_{\theta'} (x) \rvert \label{eqn:TVsum}
    \end{align}
    $$

    where the lower case $p$ denotes the PMF. 

* If we are computing the total variation distance between say a Gaussian and an exponenial which are defined over different sample spaces, then I just take the union of the sample space and set the appropriate ones to 0. What if I am interested in calculating the difference between a Bernoulli and a Gaussian? Then there is no way around calculating all possible subsets of the sample space and then actually finding the maximum. There is a trick that coming below to do this kind of thing. 

* Note that for continuous probability distributions, we just replace the sum with integral, where the integral runs over the union of the sample spaces of the two distributions in question (just appropriately set the distribution to zero when $x$ takes values outside it's domain). 

* Can you think about the graphical interpretation of the total variation distance between two continuous PDFs as areas under the curve? Reason this out. 

* Some properties of the total variation distance:

    $$
        \begin{align}
            \textrm{symmetry: } & TV(\mathbb{P}_ \theta, \mathbb{P}_ \theta') = TV(\mathbb{P}_ \theta, \mathbb{P}_ \theta') \\
            \textrm{non-negative: } & 0 \leq TV(\mathbb{P}_ \theta, \mathbb{P}_ \theta') \leq 1 \\  
            \textrm{definite: If } &  TV(\mathbb{P}_ \theta, \mathbb{P}_ \theta') = 0 \textrm{ then }\mathbb{P}_ \theta = \mathbb{P}_ \theta' \\
            \textrm{triangle inequality: } & TV(\mathbb{P}_ \theta, \mathbb{P}_ \theta') \leq TV(\mathbb{P}_ \theta, \mathbb{P}_ \theta'') + TV(\mathbb{P}_ \theta'', \mathbb{P}_ \theta')
        \end{align}
    $$

    These four axioms are necessary for the notion of a distance, much like Cartesian distances. This is a distance between probability distributions.

* Most of the calculations of Total Variation Distance involved the formula given in equation \ref{eqn:TVsum} (or its integral equivalent for continuous distributions). 

* Worked example: Find $$TV(2\sqrt{n}(\bar{X}_n - 1/2), Z)$$ where $$X_i \sim \textrm{Ber}(0.5)$$ and $$Z \sim N(0,1)$$. The trick here, and perhaps one in general is to realize that if we can find some $$A \subset E$$ for which the total variation is 1, we know that this is the worst case and hence the maximum is 1. So this involves finding a subset of the sample space $A$ for which the total variation is one, and then using the primary definition:

    $$
    \begin{align}
        TV(\mathbb{P}_ \theta, \mathbb{P}_ \theta') = \max\limits_{A \subset E} \lvert \mathbb{P}_ \theta (A) - \mathbb{P}_ \theta' (A) \rvert
    \end{align}
    $$

    To go about this, we first introduce some terminology: the support of a function $f$ is the set of all $x \in D$ for which $f(x) \neq 0$, where $D$ is the domain of the function. So, let us start the problem by finding a support for $$2\sqrt{n}(\bar{X}_n - 1/2)$$. We know that $\bar{X}_n \in \\{1/n, 2/n,\ldots,n/n\\}$ and hence $2\sqrt{n}(\bar{X}_n - 1/2) \in \\{2\sqrt{n}(k/n -1/2): k = 0, 1,\ldots,n\\}$. Now remember that we are looking for a subset of the sample space such that the $$\mathbb{P} _\theta = 1$$ and $$\mathbb{P} _\theta' = 0$$. One obvious choice for $$A \subset E$$ is to take the support itself, because necessarily for the first rv, the $$\mathbb{P} _ \theta(A) = 1$$. Furthermore, because the normal rv is continuous, the probability that it takes on values in a dicrete domain is 0, and hence $$\mathbb{P} _ \theta'(A) = 0$$. This is the worst case (i.e. the maximum) and it follows that the total variaion in this case is 1. 

* The above example exemplifies the problem with the total variation distance: the fact that the TV distance quantifies that every discrete distribution deviates to the same extent from every continuous distribution is not very informative. For example, if I were to look at the distance between $X$ and $X + a$ where $X$ is Bernoulli and $a \subset (0, 1)$. Because this is a closed interval, the TV is always 1, regardless of the magnitude of $a$. 

* Another good metric to quantify how far apart two distribution are is the **Kullback-Leibler Divergence**. The KL divergence between two pdfs (replace the integral with a sum for the discrete case) is given by

    $$
    \begin{align}
        KL(\mathbb{P} _ \theta, \mathbb{P} _ \theta') = \int_x p_\theta(x) \ln\left(\dfrac{p_\theta (x)}{p_\theta'(x)}\right)
    \end{align}
    $$

    The definition above just says that the KL distance is the expectation of $$\ln(p _ {\theta ^ *}/p _ \theta)$$. That is,
    
    $$
    \begin{align}
        KL(\mathbb{P} _ {\theta ^ *}, \mathbb{P} _ \theta) = \mathbb{E}\left[\ln\left(\dfrac{p _ {\theta ^ *} (x)}{p_\theta(x)}\right)\right] \label{eqn:KLdefn}
    \end{align}
    $$

    Note that some of the properties that we defined above for a distance metric metric do not apply for the KL divergence (hence the nomenclature of it being a divergence and not a distance). Specifically the properties are,

    * It is not symmetric
    * It is positive. There is a nice proof using Jensen's inequality for the concave ln function. Can you work this out?
    * It is definite. i.e. if the KL divergence is 0, then $$\mathbb{P} _ \theta = \mathbb{P} _ \theta'$$. This is really important because if we doing minimizations and maximizations, then we can identify a single optimum. This lets us find the value of $\theta ^ *$ s.t. $\theta ^ * = \theta$. More on minimizations coming later.
    * The triangle inequality in general does not apply. 

    A couple of points here. Note that because it is not symmetric, the order in which you calculate the divergence is important. In general, let's say that we have some true distribution $$\mathbb{P} _ {\theta ^ *}$$ that generated my data, and I am trying to estimate it with the distribution $$\mathbb{P} _ {\theta}$$. In this case, we will define the KL divergence to be $$KL(\mathbb{P} _ {\theta ^ *}, \mathbb{P} _ {\theta})$$. The second point is the definite property. If we can find some $\theta$ for which $$KL(\mathbb{P} _ {\theta ^ *}, \mathbb{P} _ {\theta}) = 0$$, it means that we have found $\theta ^ *$. In practice it is hard to make it exactly zero, and we will instead find the $$\theta$$ at which the KL divergence is a minimum. We are trying to get as close to the true $$\theta ^ *$$ as possible. 

* The positiveness of the KL divergence turns out to be super useful while estimating the distribution given some data. This is important and really practical so pay attentioin! Consider again the definition given in Equation \ref{eqn:KLdefn}. We expand the logarithm to find (the expositon below is for a pmf but a very similar argument holds for a pdf)

    $$
    \begin{align}
        KL(\mathbb{P} _ {\theta ^ *}, \mathbb{P} _ \theta) & = \mathbf{E}[\ln p _ {\theta ^ *}(x) - p _ \theta(x)] \\
        & = \mathbf{E}[\ln p _ {\theta ^ *}(x)] - \mathbf{E}[\ln p _ \theta(x)]
    \end{align}
    $$

    Now, because $$\theta ^ *$$ is a fixed (unknown) number, the expectation is just a constant $C$ and hence the above can be rewritten as 

    $$
    \begin{align}
        KL(\mathbb{P} _ {\theta ^ *}, \mathbb{P} _ \theta) & = C - \mathbf{E}[\ln p _ \theta(x)]
    \end{align}
    $$

    Now we use the most useful hammer in our toolbox: from the law of large numbers, we can replace the expectation in the above equation with a sample mean, and hence we can rewrite the equation as
    
    $$
    \begin{align}
        \hat{KL}(\mathbb{P} _ {\theta ^ *}, \mathbb{P} _ \theta) & = C - \dfrac{1}{n}\sum\limits _ {i=1} ^ n \ln (p _ \theta (X_i))
    \end{align}
    $$
    
    where the $$X _ i$$ are the observations. Note that because we now put in the observations, the KL divergence now becomes an estimator and we now need the hat symbol above it. Now, from the definiteness property (i.e. if $$KL(\mathbb{P} _ {\theta ^ *}, \mathbb{P} _ \theta) = 0$$, then $$\theta ^ * = \theta$$), we ideally want to find a $$\theta$$ as close to $$\theta ^ *$$ as possible. Because the KL divergence is always positive, the ideal number would be 0, but in practice, we minimize the KL divergence. This means that we want to find

    $$
    \begin{align}
        & \min\limits _ {\theta \in \Theta} C - \dfrac{1}{n}\sum\limits _ {i=1} ^ n \ln (p _ \theta (X_i)) \\
        \Rightarrow & \max\limits _ {\theta \in \Theta} \dfrac{1}{n}\sum\limits _ {i=1} ^ n \ln (p _ \theta (X_i)) \\
        \Rightarrow & \max\limits _ {\theta \in \Theta} \ln (\prod\limits _ {i=1} ^ n p _ \theta (X_i)) \\
        \Rightarrow & \max\limits _ {\theta \in \Theta} \prod\limits _ {i=1} ^ n p _ \theta (X_i) \\
    \end{align}
    $$

    Note that the quantity in the final equation is exactly the likelihood that I would see the observations $X_1 = x_1,\ldots,X_n = x_n$ under the model $$\mathbb{P} _ \theta$$. So in essence, we want to find that value of $\theta$ that maximizes the probability of seeing those observations occur. We can therefore define a likelihood function $$L(x_1, x_2, \ldots, x_n; \theta)$$

    $$
    \begin{align}
        L: &  E^n \times \Theta \longrightarrow \mathbb{R} \\
           & {x_1,\ldots,x_n; \theta} \mapsto \mathbb{P} _ \theta (X_1=x_1,\ldots,X_n=x_n)
    \end{align}
    $$

### Lecture 9: Introduction to Maximum Likelihood Estimation

* Let us look at an example using Bernoulli trials:

    $$
    \begin{align}
    L(x_1,\ldots,x_n; p) = p ^ {\sum x_i} (1- p) ^ {n - \sum x_i}
    \end{align}
    $$

* An example using Poisson distributed data:

    $$
    \begin{align}
    L(x_1,\ldots,x_n; \lambda) & = \prod\limits _ {i = 1} ^ n \dfrac{e ^ {-\lambda} \lambda ^ {x_i}}{x_i!} \\
    & = \dfrac{e ^ {-n\lambda}\lambda ^ {\sum x_i}}{x_1 ! \ldots x_n!}
    \end{align}
    $$

    You can similarly derive the likelihood functions for Gaussians, Exponentials, etc. 

* One point to note is that sometimes we have to use indicator functions appropriately while defining the likelihood function (because if the obervation was outside the range of the underlying distribution, the likelihood that the obervations is described by that distribution is zero). Take for example the uniform distribution $$\mathcal{U}([0, b])$$. The likelihood function is defined as

    $$
    \begin{align}
        L(x_1, \ldots, x_n) & = \dfrac{1}{b^n} \prod\limits_{i=1}^{n}\mathbb{I} \{x_i > 0\}\cdot\mathbb{I} \{x_i < b\} \\
        & = \dfrac{1}{b^n} \prod\limits_{i=1}^{n}\mathbb{I} \{\min x_i > 0\} \cdot\mathbb{I} \{\max x_i < b\} \\
        & = \dfrac{1}{b^n} \prod\limits_{i=1}^{n}\mathbb{I} \mathbb{I} \{\max x_i < b\}
    \end{align}
    $$

    Note that we omitted the requirement $$\min x_i > 0$$ because we assumed that comes naturally from the fact that we assumed a uniform distribution. If this inequality was not respected, something is definitely wrong with our assumption. 

* We often want to maximize the likelihood in the maximum likelihood estimator, but this is the same as trying to maximize the log likelihood because the log is an increasing function. This makes our life much easier in the face of a whole bunch of powers and exponentials, which often appears in the likelihood functions. **Note that we are maximizing with respect to the parameters, not the** $$\mathbf{x_i}$$. 

* Notation: The maximum likelihood estimator is denoted by $$\hat{\theta} ^ {MLE} _ n$$, and the definition is (for a pmf, but an identical definition holds for a pdf):

    $$
    \begin{align}
        \hat{\theta} ^ {MLE} _ n = \textrm{argmax} _ {\theta \in \Theta} \prod\limits_{i=1}^n p_\theta(x_i)
    \end{align}
    $$

* Now that we are interested in finding maxima of a function, we are enetering the realm of calculus, in fact, multivariable calculus. Very brief notes in thid section due to my familiarity with multivariable calculus:

    * If we have a function $f(\theta)$, where $\theta = (\theta_1, \theta_2,\ldots, \theta_n)^T$ is a column vector, then to calculate if the function in $d$-dimensional space is concave or convex I am interested in finding the Hessian Matrix $\mathbf{H}f(\theta)$, which is deifined as

        $$
        \begin{align}
            \mathbf{H}f(\theta) = \left(
            \begin{matrix}
                \dfrac{\partial^2 f}{\partial \theta_1 \theta_1} & \dfrac{\partial^2 f}{\partial \theta_1 \theta_2} & \dots & \dfrac{\partial^2 f}{\partial \theta_1 \theta_d} \\
                \dfrac{\partial^2 f}{\partial \theta_2 \theta_1} & \dfrac{\partial^2 f}{\partial \theta_2 \theta_2} & \dots & \dfrac{\partial^2 f}{\partial \theta_2 \theta_d} \\
                \vdots & \vdots & \ddots & \vdots \\
                \dfrac{\partial^2 f}{\partial \theta_d \theta_1} & \dfrac{\partial^2 f}{\partial \theta_d \theta_2} & \dots & \dfrac{\partial^2 f}{\partial \theta_d \theta_d}
            \end{matrix} \right) \label{eqn:Hessian}
        \end{align}
        $$
    
        If $x^T \mathbf{H}f(\theta)x \leq 0$ (resp. $\geq 0$), for all $x \in \mathbb{R}^d$, then $f(\theta)$ is concave (resp. convex). This is a strong requirement. This requirement becomes clearer when thinking about saddle points, etc. 

    * A symmetric real-valued $d \times d$ matrix $$\mathbf{A}$$ is positive semi-definite if $$x^T\mathbf{A}x \geq 0$$ for all $$x \in \mathbb{R}^d$$. Similarly we define negative semi-definite. If the inequality holds strictly, we use positive deinite and negative definite respectively.

    * Another way of checking for positive semi-definiteness: A symmetric real-valued $d \times d$ matrix $$\mathbf{A}$$ is positive semi-definite is all of its eigen-values are non-negative. If they are strictly positive, then $$\mathbb{A}$$ is positive definite. Similarly you can define negative (semi) definite matrices. So if all the eigenvalues are strictly negative, the function is strictly concave, and vice versa. 

    * There is a quick way to check if the eigen values are negative. If a $$2 \times 2$$ matrix $$\mathbf{A}$$ is negative semi-definite, then we need both the eigen values $\lambda_1, \lambda_2$ to be negative. This would mean that $$\textrm{tr}(\mathbf{A}) \leq 0$$ and $$\textrm{det}(\mathbf{A}) \geq 0$$. In the $$ 2 \times 2$$ case, this is easy to see because $$\textrm{tr}(\mathbf{A}) = \lambda_1 + \lambda_2$$ and $$\textrm{det}(\mathbf{A}) = \lambda_1\lambda_2$$. 

* General procedure to check if a multidimensional function is concave or convex: First calculate the Hessian for the function, given by equation \ref{eqn:Hessian}. We then need to check if the Hessian is positive (semi)definite or negative (semi)definite (convex and concave respectively). One way to do this to check the definition directly and see if $$x^T \mathbf{A}x$$ is greater than or less than zero, for all $x \in \mathbb{R}^d$. In some cases it might be easier to check the eigenvalues. If all eigenvalues are (non-negative)positive, then the matrix is positive (semi)definite, and vice versa. 

### Lecture 10: Consistency of MLE, covariance matrices, multivariate statistics

* It turns out that the maximum likelihood estimator is consistent under some mild regulaity conditions (that the pdf is continuous almost everywhere as a function of $$\theta$$), that is,

    $$
    \begin{align}
        \hat{\theta}^{MLE} \xrightarrow[n \to \infty]{\mathbf{P}} \theta^*
    \end{align}
    $$

    This works even for multivariate distributions, where $\theta = (\mu\ \sigma^2)^T$ for example. The definition of consistency for a random vector is slightly more involved and we need some deinitions here. A random vector $$\mathbf{X} = (X^{(1)},\dots, X^{(d)})^T$$ is a function that maps $$\omega \in \Omega$$ to $$\mathbf{R}^d$$, where $$\Omega$$ is the sample space of the random variables $$X^{(i)}$$, which are themselves scalar. That is,

    $$
    \begin{align}
        X:& \Omega \xrightarrow[]{} \mathbf{R}^d \\
          & \omega \mapsto \left(\begin{matrix}
                                    & X^{(1)}(\omega) \\
                                    & X^{(2)}(\omega) \\
                                    &\vdots\\
                                    & X^{(d)}(\omega)
                                \end{matrix}\right)
    \end{align}
    $$

    Note:  in general a random vector collects the outcome of the oberving $d$ ransom variables once, but we could also intepret it (depending on the situation) as $d$ independent realizations of some underlying distributioin $X$. Thinking about the former, this means that $$X^{(1)} \sim N(0,1), X^{(1)} \sim \mathrm{Poiss(\lambda)}$$, etc. For example, I could look at one person and the different components could be height, weight, income, etc. 
    
    Now assume that I have a sequence of random vectors $$\mathbf{X}_1, \ldots, \mathbf{X}_n$$ and another random vector $$\mathbf{X}$$ , then we say that 
    
    $$
    \begin{align}
        \mathbf{X}_n \xrightarrow[n \to \infty]{\mathbf{P}} \mathbf{X}
    \end{align}
    $$

    if and only if
    $$X _ n ^ {(k)} \xrightarrow[n \to \infty]{\mathbf{P}} X ^ {(k)}$$ for all $$1 \leq k \leq d$$. In other words, every component of the sequence of vectors should converge to the limiting vector. The notion of a CDF for a random vector also applies and it is defined as a function $$F$$ such that

    $$
    \begin{align}
        F: & \mathbf{R}^d \xrightarrow[]{} [0, 1] \\
           & \mathbf{x} \mapsto \mathbf{P}(X^{(1)} \leq x^{(1)},\ldots,X^{(1)} \leq x^{(1)})
    \end{align}
    $$

* In general, $$\textrm{Cov}(X,Y) = 0$$ does not imply that $$X$$ and $$Y$$ are independent. This is only true if $$(X, Y)^T$$ form a Gaussian vector. What is a Gaussian vector? **It doesn't mean that** $$X$$ **and** $$Y$$ **are both Gaussian.** It means something more, it means that any linear combination $$\alpha X + \beta Y$$ is also Gaussian, where $(\alpha, \beta) \in \mathbb{R}^2 \ (0,0)$. Can you show this using the example $$X \sim N(0,1)$$ and $$Y \sim R \cdot X$$, where $$R = 2B -1$$, with $$B \sim \textrm{Ber}(1/2)$$. In other words, $$R$$ (called the Radhemacher random variable) flips the sign of $$X$$ with probability 1/2. Take the linear combination $$X+Y$$. What happens here?

* When talking about a random vectors (remember that here we just say that $$\mathbf{X} = (X^{(1)}\ X^{(2) \ldots X^{(d)}})^T)$$, it becomes extremely useful to begin to think of covariance matrices which is defined by

    $$
    \begin{align}
        \mathbb{C}\textrm{ov}(\mathbf{X}) _ {ij} = \sigma _ {ij} = \textrm{Cov}(X^{(i)}, X^{(j)})
    \end{align}
    $$

    It just calculates the pairwise covariances between every component of the random vector $$\mathbf{X}$$. This is just convenient notation that also allows us to make a lot of useful transformations and manipulations as we will see later. Just think of it as compactly finding pairwise covariances of the components of a random vector and arranging them into a matrix. Here are some properties of the covariance matrix:

    * The diagonal entries $$ii$$ are just the variance of $$X^{(i)}$$. 
    * Let $$X$$ be a random vector of dimensions $$d \times 1$$, and let $$\mathbf{A}$$ be a $$n \times d$$ matrix, and let $b$ be a $$n \times 1$$ column vector. Then under the affine transformation $$\mathbf{AX} + b$$, $$\mathbb{C}\textrm{ov}(\mathbf{AX + b}) = \mathbf{A}\mathbb{C}\textrm{ov}(\mathbf{X})\mathbf{A}^T$$.
    * $$\Sigma$$ is positive definite, so it is diagonalizable.
    * $$\Sigma$$ is positive definite, so there exists an orthogonal matrix $$U$$ such that $$D = U\Sigma U^T$$, where $$D$$ is a diagonal matrix. (For an orthogonal matrix $$U$$, $$UU^T = U^TU = I$$). $$D$$ contains the eigenvalues of $$\Sigma$$. 
    * $$\Sigma$$ has a unique square too i.e. there exists a matrix $$\Sigma^\frac{1}{2}$$ such that  $$\Sigma^\frac{1}{2} \cdot \Sigma^\frac{1}{2} = \Sigma$$. 
    * $$\Sigma$$ is positive definite, so there exists a diagonal matrix $$D = U\Sigma U^T$$ that has entries that are all strictly positive, then it is invertible and the inverse $$\Sigma^{-1}$$ satisfies the following: $$\Sigma^{-\frac{1}{2}} \cdot \Sigma^{-\frac{1}{2}} = \Sigma^{-1}$$, where $$\Sigma^{-\frac{1}{2}}$$ is the inverse square root of $$\Sigma$$. 

* Now, we are ready to define Gaussian random vectors and the multivariate Gaussian distribution. A vector $$\mathbf{X} = (X^{(1)}\ X^{(2)})^T$$ is a Gaussian vector if any linear combinations of the $$X^{(i)}$$ is also Gaussian i.e. $$\alpha^T\mathbf{X}$$ is Gaussian for any $$\alpha \in \mathbb{R}^d$$. For a multivariate Gaussian distribution, I only need to define the expectation vector $$\mu = \mathbf{E}[ \mathbf{X} ]$$ and the covarianc matrix $$\Sigma = \mathbb{C}\textrm{ov}(\mathbf{X})$$. Then the pdf of the multivariate Gaussian distribution is

    $$
    \begin{align}
        f(\mathbf{X}) = f(X^{(1)}, \ldots, X^{(d)}) = & \left(\dfrac{1}{(2\pi)\textrm{det}(\mathbf{\Sigma)}}\right)^{d/2} \cdot \\
        & \exp\left[-\dfrac{1}{2} (x-\mu)^T\Sigma^{-1}(x-\mu)\right]
    \end{align}
    $$

* Armed with this knowledge, we can now extend the central limit theorem to $d$ dimensions. The formulation in this case look very similar to the case of a scalar random variable except that we replace all the scalars with Gaussian vectors (remember the conditions on what makes something a Gaussian vector?). The central limit theorem in $d$ dimensions says that if $$\mathbf{X} \in \mathbb{R}^d$$ is a random vector with mean $$\mathbf{E}[\mathbf{X}] = \mu$$ and covariance matrix $\Sigma$. Let $$\mathbf{X}_1, \ldots, \mathbf{X}_1$$ be i.i.d copies of $$\mathbf{X}$$. Then,   

    $$
    \begin{align}
        \sqrt{n}\Sigma^{-1/2}(\bar{\mathbf{X}}_n - \mathbf{E}[X])\xrightarrow[n \to \infty]{(d)} N_d(0, I_d)
    \end{align}
    $$

    where the expectation is defined somewhat intuitively - take the expectation of every single component in the vector, $\Sigma^{-1/2}$ is the square root of the covariance matrix, and $I_d$ is the identity matrix.

* In addition to having the central limit theorem in the case of random vectors of $d$ dimensions, we also have the multivariate delta method, which generalizes the delta method. Consider a sequence of random vectors $$\mathbf{T}_n in \mathbb{R}^d$$. Assume that the central limit theorem applies, i.e.,

    $$
    \begin{align}
        \sqrt(n)(\mathbf{T}_n -\theta) \xrightarrow[n \to \infty]{(d)} N_d(0, \Sigma)
    \end{align}
    $$

    for some $$\theta \in \mathbb{R}^d$$. Now consider I have a function $$g: \mathbb{R}^d \mapsto \mathbb{R}^k, k \geq 1$$,  and if $$g(\theta)$$ is continuously differentiable, then the multivariate delta method states that

    $$
    \begin{align}
        \sqrt(n)(g(\mathbf{T}_n) -g(\theta)) \xrightarrow[n \to \infty]{(d)} N_d(0, (\nabla g(\theta))^T \Sigma \nabla g(\theta))
    \end{align}
    $$

    Some notes on taking the gradient of a vector function $\nabla g(\theta)$. We know that $$g = g(\theta_1, \ldots, \theta_d)$$ lives in $$\mathbb{R}^k$$, and hence has $$k$$ components $$(g_1(\theta_1, \ldots, \theta_d), \ldots, g_k(\theta_1, \ldots, \theta_d))$$. Therefore, we can think of the gradient of this function taking the gradient of each component $$g_j$$ and stacking them beside each other. The gradient of each component is itself a column vector in $$\mathbb{R}^d$$. In other words,

    $$
    \begin{align}
        (\nabla g(\theta))_{ij}  = \left(\dfrac{\partial g(\theta)_j}{\partial \theta_i}\right)_{1 \leq i \leq d, 1 \leq j \leq k} \in \mathbb{R}^{d \times k}
    \end{align}
    $$

    For example, consider taking the gradient of

    $$
    \begin{align}
        f(x, y, z) = \left(
                            \begin{matrix}
                                &x^2 + y^2 \\
                                & 2xyz \\
                                & z^2 + 4\\
                                & xy + \sin(z)
                            \end{matrix}
                    \right)
    \end{align}
    $$

    Here $$\theta = (x, y, z)^T$$ and $$f$$ has four components. Therefore $$k \in {1,2,3,4}$$ and $$i \in {1,2,3}$$. The transpose of the gradient matrix is also called the **Jacobian matrix**. 

 * Recitation 4 in unit 3 taught me something quite nice about when and why we need to use the multivariate Gaussian distribution during parameter estimation. Let's say that we have a Gaussian of unknown mean and variance $N(\mu, \sigma^2)$. It is easy enough to construct an estimator for the mean $\mu$ because the law of large numbers dictates that it is just the sample mean. However, finding an estimator for the variance is a little bit more involved. It is natural to assume that given a series of observations $X_1, \ldots, X_n$, we can assume that a reasonable estimator is (see how sample means are your best friend?!)

    $$
    \begin{align}
        \hat{\sigma^2} = \dfrac{1}{n}\sum\limits_{i=1}^nX_i^2 - \left(\dfrac{1}{n}\sum\limits_{i=1}^n X_i \right)^2
    \end{align}
    $$

    This is reasonable because by the LLN, we would expect the first term to converge to $$\mathbf{E}[X_1^2]$$ and the second term to converge to $$(\mathbf{E}[X_1])^2$$ (after an application of the continuous mapping theorem as well actually). But because these two terms clearly have non-zero covariance, if I want to find confidence intervals for my estimator, to find the variance of the normal distribution that it converges to involved accounting for the fact that the two terms ae covariant. That is, we can say that the vector 

    $$
    \begin{align}
        T_n = \left( \begin{matrix}
                    & \dfrac{1}{n}\sum\limits_{i=1}^nX_i^2 \\
                    & \dfrac{1}{n}\sum\limits_{i=1}^n X_i
                \end{matrix} 
        \right) 
        \xrightarrow[n \to \infty]{\mathbf{P}} 
        \left( \begin{matrix}
                    & \mathbf{E}[X_1^2] = \sigma^2 + \mu^2 \\
                    & \mathbf{E}[X_1] = \mu
                \end{matrix} 
        \right) = \theta,
    \end{align}
    $$

    and we can apply the central limit theorem on that vector to obtain

    $$
    \begin{align}
        \sqrt{n}(T_n - \theta) \xrightarrow[n \to \infty]{(d)} N_2(0, \Sigma)

    where $\Sigma$ is the covariance matrix between $X_1$ and $X_1^2$. So to properly apply the CLT, we need to talk about the multivariate CLT, and properly account for the covariances between the components of the matrix that is converging. But when we want to find the estimator, what we are doing is taking that vector and feeding it into a function $$g: \mathbb{R}^2 \mapsto \mathbb{R}$$ that takes the components of this vector and spits out a number. Specifically, in this case the function is $$g(x, y) = x - y^2$$. Applying this function to $$T_n$$ correctly outputs my estimator of $$\sigma^2$$, and hence the delta method applies here:

    $$
    \begin{align}
        \sqrt{n}(g(T_n) - g(\theta)) \xrightarrow[n \to \infty]{(d)} N_2(0, (\nabla g(\theta))^T \Sigma (\nabla g(\theta)))
    \end{align}
    $$

    where $x$ and $y$ in the function $g$ represent the $x$ and $y$ (i.e. the appropriate) components of $\theta$. Carrying out this calculation gives us the correct variance of the distribution that my scaled and centered estimator converges to,  and hence I can find the correct confidence intervals for my estimator (which is a function of dependent random variables, but remember that the observations are themselves independent from each other). 

### Lecture 11: Fisher Information, Asymptotic Normality of MLE, Method of Moments

* Let us go begin by going back to likelihood functions. The likelihood function $$L_1(x; \theta),\ \theta \in \mathbb{R}^d$$ is just a different notation for the pdf $$f_\theta (x)$$. Define the log-likelihood as $$l_1(\theta) = \ln(L_1(x; \theta))$$. Now, can calculate the gradient of $$l_1(\theta)$$, and because $$\theta \in \mathbb{R}^d$$,  $$\nabla l_1(\theta) \in \mathbb{R}^d$$. We have already defined the notion of a covariance matrix for a $d$ dimensional vector and hence we can calculate

    $$
    \begin{align}
        \mathbb{C}\textrm{ov}(\nabla l(\theta)) = \mathbf{E}[\nabla l(\theta)(\nabla l(\theta))^T] - \mathbf{E}[\nabla l(\theta)]\mathbf{E}[\nabla l(\theta)]^T \equiv I(\theta)
    \end{align}
    $$

    where by definition we define this covariance matrix of the gradient of the log-likelihood function to be the **Fisher Information** $$I(\theta)$$. Now there is a theorem that states that 

    $$
    \begin{align}
        I(\theta) = -\mathbf{E}[\mathbf{H}l(\theta)]
    \end{align}
    $$

    where $$\mathbf{H}$$ denotes the Hessian matrix as before. This is somewhat remarkable: the covariance matrix is related to an expected value. For the 1D case, the covariance matrix is just $$1 \times 1$$, and hence $$\nabla l(\theta) = l'(\theta)$$,and  $$I(\theta) = \textrm{Var}(l'(\theta))$$. Then, from the theorem above, we have that

    $$
    \begin{align}
        \textrm{Var}(l'(\theta)) = -\mathbf{E}[l''(\theta)]
    \end{align}
    $$

* So why go through all this singing and dancing for the Fisher information? It turns out that it has implications for the aymptotic normality of the MLE estimator. Under some fairly mild conditions (see the lecture - it accounts for the standard conditions such as identifiability, the support of $$\mathbb{P} _ \theta$$ does not depend on $$\theta$$, and importantly, that the Fisher information matrix is invertible in a region around $$\theta ^ *$$), we can show that the MLE estimator is both consistent and asumptotically normal. In other words, 

    $$
    \begin{align}
        & \hat{\theta}_n^{MLE} \xrightarrow[n \to \infty]{\mathbf{P}} \theta^*  \mathrm{w.r.t.\ \mathbb{P} _ {\theta^*}}\\
        & \sqrt{n}(\hat{\theta}_n^{MLE} - \theta^ * ) \xrightarrow[n \to \infty]{(d)} N_d(0, I^{-1}(\theta ^ * )) \mathrm{w.r.t.\ \mathbb{P} _ {\theta^*}}
    \end{align}
    $$

    where $$\theta^ *$$ is the true parameter. There is an outline of a proof in the lecture videos, but the detailed proof is beyond the scope of this class. The key idea is that the likelihood function is maximized near $$\theta ^ *$$ and hence the slope of $$l(\theta$)$$ and the expected value of the slope is 0. 

    **The whole point of the Fisher information is that helps us determine the variance of the normal distribution that that** $$\hat{\theta}_n^{MLE} - \theta ^ *$$ **converges to. So to find confidence intervals, for example, we need to write out the log-likelihood function for one observation** $$l(\theta)$$, **find the Fisher information** $$I(\theta)$$ **at** $$\theta ^ *$$,  **and find it's inverse. Intuitively, the more "information" I have, the tighter the variance in my estimator.**   

**Method of moments**

* Assume that I have some statiscal model $$(E, \mathbb{P} _ {\theta, \theta \in \Theta})$$ and $$\Theta \subset \mathbb{R}^d$$, for some $d \geq 1$$. Then I define the moment of this distribution as being

    $$
    \begin{align}
        m_k(\theta) = \mathbf{E}[X_1^k]
    \end{align}
    $$

    Now the law of large numbers tells us that (for the method of moments, we only consider $$k = 1, \ldots, d$$)

    $$
    \begin{align}
        \hat{m} _ k = \dfrac{1}{n}\sum\limits _ 1 ^ n X_i ^ k \xrightarrow[n \to \infty]{\mathbf{P}, \mathrm{a.s.}} m_k(\theta)
    \end{align}
    $$

    where $$\hat{m} _ k $$ are called the emperical moments. More compactaly, we can say that the LLN tells us 

    $$
    \begin{align}
        (\hat{m} _ 1, \ldots, \hat{m} _ d \xrightarrow[n \to \infty]{\mathbf{P}, \mathrm{a.s.}} (\hat{m} _ 1(\theta), \ldots, m _ d(\theta))
    \end{align}
    $$

    Now we define a function $$M: \mathbb{R} ^ d \mapsto \mathbb{R} ^ d$$, where it takes the $$d$$ dimensional vector $$\theta$$ and maps it to the vector of its first $$d$$ moments, i.e., 

    $$
    \begin{align}
        M(\theta) = (m_1(\theta), \ldots, m_d(\theta))
    \end{align}
    $$
    
    For example, for a Gaussian, $$M(\mu, \sigma) = (m_1(\mu, \sigma), m_2(\mu, \sigma)) = (\mu, \mu^2 + \sigma^2)$$. 
    Assuming that $$M$$ is one to one, $$\theta = M^{-1}(m_1(\theta), \ldots, m_d(\theta))$$. Then, by definition, the moments estimator of $\theta$$ is given by

    $$
    \begin{align}
        \hat{\theta}_n^{MM} = M^{-1}(\hat{m}_1, \ldots, \hat{m}_1)
    \end{align}
    $$

    provided it exits. 

* Note that choosing the first $d$ moments is somewhat arbitrary. The goal here is to integrate out the values of the random variables (specifically through expectations because then we can replace the expectation with a sample mean by the LLN) so that I have $d$ different equations for the $d$ parameters. So in general if $$\theta \in \mathbb{R}^d$$, we only need to define the expectation vector to be $$M(\theta) = (g_1(\theta), \ldots, g_d(\theta))$$ and then our estimator would consist of the vector that calculates the sample mean of the values of the function. It is just so happens that the $k$-th moment is often convenient. 

* The choice of the functions being the $$k$$-th moments is somewhat arbitrary. We are really looking for a set of functions that gives us a set of $$d$$ equations that we can use to solve for each of the components of $$\theta \in \mathbb{R}^d$$. More generally, we could have written our a function 

    $$
    \begin{align}
        M(\theta) = (g_1(X), \ldots, g_d(X))
    \end{align}
    $$

    and then use our hammer of replacing expected values with sample means to obtain an estimator of the form 

    $$
    \begin{align}
        \hat{M} = (\frac{1}{n}\sum_i g_1(X_i), \ldots, \frac{1}{n}\sum_i g_d(X_i))
    \end{align}
    $$

    Now, we know from an application of the central limit theorem that

    $$
    \begin{align}
        \sqrt{n}(\hat{M} - M(\theta)) \xrightarrow[n \to \infty]{(d)} N_d(0, \Sigma)
    \end{align}
    $$

    where $$\Sigma = \mathbb{C}\mathrm{ov}((g_1(X_1)), \ldots, g_d(X_1)^T)$$ is the covariance matrix. We can now apply the multivariate delta method to find the actual estimator. We are interested in findging $$M^{-1}(M(\theta))$$, which means we need to use the delta method. Therefore, we have after using the delta method that

    $$
    \begin{align}
        \sqrt{n}(M^{-1}(\hat{M}) - M^{-1}(M(\theta))) \xrightarrow[n \to \infty]{(d)} N_d(0, (\nabla M^{-1}(M(\theta))^T\Sigma \nabla M^{-1}(M(\theta))
    \end{align}
    $$

### Lecture 12: M-Estimation 

* M-Estimation is very similar to log-likelihood in that it generalizes the idea behind the log-likelihood. The idea here is that let's say we have a function $$\rho(X_1, \mu)$$ and I want to find the value of $\mu$ that maximizes (or minimizes, depending on my goal and the specfic nature of my function $$\rho$$) the quantity $$\mathbf{E}[ \rho(X_1,\mu)] $$. That is, we want to find a function $$\rho: E \times \mathcal{M} \mapsto \mathbb{R}$$, where $$\mathcal{M}$$ is the set of possible values of the unknown parameter $$\mu^*$$ such that

    $$
    \begin{align}
        \mu = \underset{\mu \in \mathbb{R}^d}{\mathrm{argmin}}\mathbf{E}[\rho(X_1,\mu)]
    \end{align}
    $$

    achieves its minimum at $$\mu = \mu^*$$. 

    Note that we need to make no modeling assumption that the data is drawn from some family of distributions $$\mathbb{P}$$. 

    In general, the goal is to find the **loss function** $$\rho(X,\mu)$$ such that $$\mathcal{Q}(\mu) = \mathbf{E}[\rho(X,\mu)]$$ attains a minimum at $$\mu = \mu^ * $$. Because $$\mathcal{Q}$$ is an expectation, we can proceed by replacing this expectation with the sample mean of $$\mathcal{Q}(\mu)$$. The goal is to find a function $$\rho(X, \mu)$$ such that the expectation of that function is minimized at the parameter of interest $$\mu^ * $$. 

* **Theorem**: Assume that $$(E, {\mathbb{P} _ \theta)} _ {\theta \in \Theta})$$ is a statistical model associated with the data.  Let $$\mathbb{M} = \Theta$$ and $$\rho(x, \theta)  = -L_1(x, \theta$$, provided the log likelihood is positive everywhere. Then $$\mu^* = \theta^* $$, that is $$\theta^* $$ is the true value of the parameter. The goal is to find a function $$\rho(X, \musuch that the expectation of that function is minimized at the parameter of interest. 

* As a special case of M-estimation, we can think about a median $$\mathrm{med}(X)$$ which is any number such that if $$X$$ is a discrete random variable,

    $$
    \begin{align}
        \mathbf{P}(X > \mathrm{med}(X)) = \mathbf{P}(X > \mathrm{med}(X)) = 1/2
    \end{align}
    $$

    It turns out that if I pick the loss function $$\rho(X,\mu) = \lvert X - \mu \rvert$$, then 

    $$
    \begin{align}
        \textrm{argmin}_{\mu \in \mathcal{M}} \mathbf{E}[\lvert X - \mu \rvert]
    \end{align}
    $$

    is the median of $$X$$. 

* Note: The expectation of a convex function is convex. This can be proved by an application of the mean value theorem (i.e. any chord lies above the curve for a convex function). 

* The strategy in M-estimation follows similar lines as in maximum likelihood estimation. There are three steps:

    0. Define a function $$\rho(X, \mu)$$ and hence the expectation of that function $$Q(\mu) = \mathbf{E}[\rho(X,\mu)]$$.
    0. Replace that expectation with a sample mean (allowed from the law of large numbers) and define your estimator
    0. The estimator is the argmin of this expectation.

* Define

    $$
    \begin{align}
        \hat{\mu} _ n = \underset{\mu \in \mathcal{M}}{\mathrm{argmin}} \dfrac{1}{n}\sum\limits _ {i=1}^n \rho(X_i, \mu)
    \end{align}
    $$
    
    where $$\mathcal{M}$$ is the set of all possible value the true parameter of interest $$\mu^*$$ can take. $$\rho$$ is chosen according to what we want. Some examples include the emperical mean, emperical median, MLE,  emperical quantiles, etc. 

    Let 
    
    $$
    \begin{align}
        J(\mu) = \mathbb{E}[\dfrac{\partial^2\rho(X_1, \mu)}{\partial \mu \partial \mu^T}]
    \end{align}
    $$

    Let 

    $$
    \begin{align}
        K(\mu) = \textrm(Cov)\left( \dfrac{\partial \rho(X_1, \mu)}{\partial \mu} \right)
    \end{align}
    $$

    Let $$\mu^ * \in \mathcal{M}$$ be the true parameter and assume the following:

    0. $$\mu^*$$ is the only minimizer of the function $$\mathcal{Q}$$
    0. $$J(\mu)$$ is invertible for all $$\mu \in \mathcal{M}$$. 
    0. A few more technical conditons

    Then , $$\hat{\mu}_n$$ satisfies

    0. $\hat{\mu} _ n \xrightarrow[n \to \infty]{\mathbb{P}} \mu^* $
    0. $\sqrt{n}(\hat{\mu}_n - \mu^ * ) \xrightarrow[n \to \infty]{(d)} N(0, (J(\mu^ * ))^{-1} K(\mu^* )(J(\mu^ * ))^{-1}) $

    Note that M-estimation is a minimiization problem. Therefore, to reduce this to the log-likelihood case, $$\rho = -l(\theta)$$. This sign is very important and has confused you before because we have so far looked at maximizing the log-likelihood. 

## Unit 4: Hypothesis Testing

Here we are looking at some more specific cases of hypothesis testing where the sample size is small and the central limit theorem does not hold. We look at some specific tests designed especially for small sample sizes. What if I was given that my data is Gaussian, but I don't know the variance, and that the sample size is small?
    
We also look at example of non-parametric tests, for example, goodness of fit tests. Is my assumption on the distribution a good assumption? Presumably this would be very widely useful.

### Lecture 13: Chi squared distribution, T-test

* To understand the case where I have only a small number of samples but I am given that the data is Gaussian, let us consider the case of a clinical trial. We want to determine if a particular drug decreases the LDL cholestrol. In practice, we generally take two groups of people: a test group who gets the drug and a control group who get a placebo. Let $$\Delta_d$$ be the decrease in the LDL level in the test group, and let $$\Delta_c$$ be the decrease in the control group. Let $$X_1, \ldots, X_n \sim \mathcal{N}(\Delta_d, \sigma_d^2)$$ and  $$Y_1, \ldots, Y_m \sim \mathcal{N}(\Delta_c, \sigma_c^2)$$, that is, we are given that the data is Gaussian (which is very often the case). Note that typically $$n>m$$ because we want to give more people at getting a chance at receving a potential drug. 

* Under these assumptions, we can now define the null hypothesis $$H_0$$ and the alternative hypothesis $$H_1$$. 

    $$
    \begin{align}
        H_0: \Delta_d = \Delta_c, H_1: \Delta_d > \Delta_c
    \end{align}
    $$

    Now, because we are given that the two distributions are Gaussian, we know that 

    $$
    \begin{align}
        \bar{X}_n \sim \mathcal{N}\left(\Delta_d, \dfrac{\sigma_d^2}{n}\right) \\
        \bar{Y}_m \sim \mathcal{N}\left(\Delta_c, \dfrac{\sigma_c^2}{m}\right)
    \end{align}
    $$

    and hence

    $$
    \begin{align}
        \dfrac{\bar{X}_n - \bar{Y}_m - (\Delta_d - \Delta_c)}{\sqrt{\sigma_d^2/n + \sigma_c^2/m}} \sim \mathcal{N}(0,1)
    \end{align}
    $$

    Note that because we were given that the data is Gaussian, there was no need to invoke the central limit theorem to show the normality of this random variable.

* Note that as before we might want to not have the unknown variance in the denominator and we want to use Slutsky to do plug-in. In this case, we have to think about the relative sizes of $$m$$ and $$n$$. If $$m$$ is 12 and $$n$$ is 12000, then we really have to be careful and have a problem. In general, if we assume that $$n = cm$$, where $$c$$ is a constant, we are typically OK. 

* Consider now that I want to formulate a test statistic for this problem. The natural choice would be

    $$
    \begin{align}
        R_\psi = \left\{\dfrac{\bar{X}_n - \bar{Y}_m - (\Delta_d - \Delta_c)}{\sqrt{\dfrac{\hat{\sigma^2_d}}{n} + \dfrac{\hat{\sigma^2_c}}{m}}} > q_\alpha \right\}
    \end{align}
    $$

    Note that this is one sided because of the choice of $H_0$: I have eliminated the case that $$\Delta_d < \Delta_c$$.

* Now to determine whether or not I accept or reject the null hypothesis based on the test statistic given above, I need to know what the variances are and therein lies the problem. The initial $X_i$ were iid $$\mathcal{N}(\Delta_d, \sigma_d^2)$$ with unknown $\Delta_d$ and $\sigma_d^2$ (and likewise for the $Y_i$, so how do I go about making this decision?) This is where the student T test and the student T distribution enters into the picture. We first need to introduce various other definitions. 

* The $$\chi_d^2$$ distribution (with $d$ degrees of freedom). Let $$Z_1,\ldots, Z_n$$ be iid $$\mathcal{N}(0, 1)$$. Further, let $$\bar{X}_n$$ represent the sample mean and let 

    $$
    \begin{align}
        S_n = \dfrac{1}{n}\sum\limits_{i=1}^n (X_i - \bar{X}_ n )^2 = \dfrac{1}{n}\left(\sum\limits_{i=1}^n X_i^2\right) - (\bar{X}_n)^2 
    \end{align}
    $$

    denote a estimator (biased) for the variance. **Conchran's theorem** then states that

    $$
    \begin{align}
        \bar{X}_n \textrm{ is independent of } S_n \\
        \dfrac{nS_n}{\sigma^2} = \chi_{n-1}^2
    \end{align}
    $$


    The proof is a good exercise in understanding orthoginal matrices, so make sure you can recall and understand (see the recitation where this is derived)

* We also define the student t distribution, which is defined by the distribution of the random variable

    $$
    \begin{align}
        \dfrac{Z}{\sqrt{V/n}}
    \end{align}
    $$

    where $Z \sim \mathcal{N}(0,1)$, $V \sim \chi^2_{n}$. Note that this distribution only depends on the degree of the $$\chi^2$$ distribution. We can now exploit this fact to solve our initial problem: how do we design tests for normally distributed data when we have small sample sizes and we cannot apply Slutsky?

* The first step is to center and scale our estimators. consider the test statistic

    $$
    \begin{align}
        T_n = \sqrt{n}\dfrac{\bar{X}_n - \mu}{\sigma}
    \end{align}
    $$

    We know that $$T_n$$ is a standard Gaussian for all $n$. But we do not know $$\sigma^2$$, and we can't use Slutsky to say that $$\hat{\sigma^2} \to \sigma^2$$ because we don't have a large enough sample size. So what we do is to consider the statistic above. Now we know that $$S_n = (n-1)/n\tilde{S}_n$$, where $$\tilde{S}_n$$ is the unbiased estimator  

    $$
    \begin{align}
        \tilde{S} _ n = \dfrac{1}{n-1}\sum\limits_{i=1}^n (X_i - \bar{X}_ n )^2 
    \end{align}
    $$

    Therefore, we now do the following manipulations. Consider the estimator

    $$
    \begin{align}
        T_n & = \sqrt{n}\dfrac{\bar{X}_n - \mu}{\sqrt{\tilde{S}_n}} \\
            & = \sqrt{n}\dfrac{(\bar{X}_n - \mu)/\sigma}{\sqrt{\tilde{S}_n/\sigma^2}}\\
            & = \dfrac{Z}{\sqrt{\tilde{S}_n/\sigma^2}}
    \end{align}
    $$

    where $$Z$$ is a standard normal. Now, by Cochran's theorem, we know that $$nS_n/sigma^2 = (n-1)\tilde{S}_n\sigma^2 \sim \chi^2 _ {n-1}$$ and hence we ca substitute this in above to get
    
    $$
    \begin{align}
        T_n & = \dfrac{Z}{\sqrt{V/(n-1)}} \sim t_{n-1}
    \end{align}
    $$
    
    where $$V \sim \chi^2_{n-1}$$. The distribution $$t_{n-1}$$ is called the Student $$t$$ distribution. This is a pivotal distribution (meaning that it doesn't depend on any parameter of interest) and hence we can look up tables as a function of the degree and now derive a test statistic. Therefore, a good test at level alpha would be

    $$
    \begin{align}
        \psi_\alpha = \mathbb{1}\left\{ T_n > q_\alpha \right\}
    \end{align}
    $$ 

    or 

    $$
    \begin{align}
        \psi_\alpha = \mathbb{1}\left\{ \lvert T_n \rvert > q _ {\alpha/2} \right\}
    \end{align}
    $$ 
    depending on whether I have a one-sided test or two-sided test respectively. Here $q_\alpha$ is the $$1 - \alpha$$ quantile of the student t distribution (not Gaussian of course). Rememeber that $$q_\alpha$$, the $$1 - \alpha$$ quantile, defined as that number such that $$\mathbf{P}(T > q_\alpha) = \alpha$$, where $$T$$ is a random variable described by the pdf of the Student t distribution. 

    The student t distribution is a bit heavier tailed than the Gaussian. This makes sense because we know less about the distribution. We don't know the variance. However for large enough $n$ (usually $$n > 40$$), we can use Slutsky to show that $t_n \sim Z$, a standard normal. 

* Now what about the two sample test we started with earlier. We could deisgn the problem as having an estimator that is a vector:

    $$
    \begin{align}
        \hat{\theta} = \left( 
                            \begin{matrix}
                                \bar{X}_n \\
                                \bar{Y}_n
                            \end{matrix}
                       \right)
    \end{align}
    $$

    and the applying the multivariate delta method (using the function $$g(x, y) = x - y$$, and the modifying the covariance matrix as $$(\nabla g(\theta_0))^T \Sigma (\nabla g(\theta_0))$$ and proceeding. $$\Sigma$$ would be diagonal because $$X_i$$ and $$Y_i$$ are independent. This would be the formal, foolproof way. But we can also proceed by just noting that because because $$X_i$$ and $$Y_i$$ are independent, we know by observation that $$\textrm{var}(\bar{X}_n - \bar{Y}_n) = \sigma_d^2/m + \sigma_c^2/m$$ and hence by replacing the $$\sigma^2$$ with the unbiased estimator $$\hat{\sigma^2} = \tilde{S}_n$$, we know that

    $$
    \begin{align}
        T_n = \dfrac{\bar{X}_n - \bar{Y}_m - (\Delta_d - \Delta_c)}{\sqrt{ \hat{\sigma_d^2}/m + \hat{\sigma_c^2}/m }} \sim t_N
    \end{align}
    $$

    and to be conservative we would choose $$N = \min(n,m)$$. But there exists a more accurate formula called the Welch-Satterthwaite formula, which says that we choose

    $$
    \begin{align}
        N = \dfrac{\sqrt{ \hat{\sigma_d^2}/m + \hat{\sigma_c^2}/m }}{ \hat{\sigma_d^4}/(n^2(n-1)) + \hat{\sigma_c^4}/(m^2(m-1)) }
    \end{align}
    $$

* Computing $$p$$ values for a statistic that is distributed according to a $$p$$ value is the same procedure. We calculate the value of the test statistic (think carefully whether we want the absolute value signs, which direction should inequalitites go - this is most intuitive. In general, we want a test to reject only if some it is large enough, after allowing for some error), and then the $$p$$ value becomes

    $$
    \begin{align}
        p = \mathbf{P}(t_N > T_N)
    \end{align}
    $$

    i.e. what is the probability that a Student t distributed random variable can take on values larger than the calculated test statistic. 

* An advantage of using the Student t test is that it is a one size fits all in the sense that, for small sample sizes, (and for normally distributed data), is the correct test to use, but for large sample sizes, we should be looking up CLT tables, but, Student t converges to CLT anyway. So no need to look up two different tables. 

* **Wald's Test**: This is a slight modification to the MLE estimator. Consider again the MLE estimator, under the hypotheses $$H_0: \theta = \theta_0, H_1: \theta \neq \theta_0$$. Then, we know that

    $$
    \begin{align}
        \sqrt{n}(I(\theta_0))^{1/2}(\hat{\theta}^{MLE} - \theta_0) \xrightarrow[n \to \infty]{(d)} \mathcal{N}_d (0, I _ d)
    \end{align}
    $$

    where $$I(\theta_0) = I(\theta^*) = I(\hat{\theta}^{MLE})$$ is the Fisher information. That last replacement comes from one of the assumptions of Wald's test. According to the test, we first take the square norm (or the L2 norm) on both sides

    $$
    \left\Vert \sqrt{n}(I(\theta_0))^{1/2}(\hat{\theta}^{MLE} - \theta_0) \right\Vert \xrightarrow[n \to \infty]{(d)} \left\Vert\mathcal{N}_d (0, I _ d) \right\Vert
    $$

    Now, we can expand that LHS using the fact that $$\Vert (\cdot) \Vert = (\cdot) ^T (\cdot)$$ and that the fisher information is a symmetric matrix (and hence $$(\cdot)^T = (\cdot)^{-1}$$) to obtain

    $$
    \begin{align}
        n\left(  (\hat{\theta} - \theta_0)^T I(\hat{\theta}) (\hat{\theta} - \theta_0) \right) \xrightarrow[n \to \infty]{(d)} \chi_d^2
    \end{align}
    $$

    where $$d$$ is the length of $$\theta$$ (or $$\hat{\theta}$$). We can now use the LHS as a test statistic $$T_n$$ and now define quantiles for the chi-squared distribution to get tests at a desired level.

* The geometric interpretation of Wald's test is that we have a vector for the estimator and vector for the true parameter (or the parameter in the hypothesis). The length of the vector joining these two vectors is a random variable, and the length is distributed according to the Chi sqaured distribution. The test then defines a level around the probability that the length of this difference vector exceeds a certain length (because it is chi squared distributed). 

* We can show that for 1D, the quantiles for the Wald's test is related to the quantiles of the standard Gaussian through $$q_\alpha(\chi^2) = (q_{\alpha/2}(\mathcal{N}(0,1)))^2$$. 

* **Constrained Maximum Likelihood estimator**: Suppose that my hypothesis has the form 

    $$
    \begin{align}
        H_0: (\theta^* _ {r+1}, \ldots, \theta^* _ {d}) & = (\theta^{(o)} _ {r+1}, \ldots, \theta^{(o)} _ {d}) \\
        H_1:  (\theta^* _ {r+1}, \ldots, \theta^* _ {d}) & \neq (\theta^{(o)} _ {r+1}, \ldots, \theta^{(o)} _ {d})
    \end{align}
    $$

    then $$\Theta_0$$, the region of the null hypothesis, is

    $$
    \begin{align}
        \Theta_0:= \left\{  \mathbf{v} \in \mathbb{R}^d: ( v_{r+1}, \ldots, v_{d} ) = (\theta_{r+1}^{(o)}, \ldots, \theta_{d}^{(o)})  \right\}
    \end{align}
    $$
    where $$\theta_{r+1}^{(o)}, \ldots, \theta_{d}^{(o)}$$ are all known. In other words, we are saying that we want to find the maximum under the constraint that those values of my estimator are fixed. Now, under the likelihood estimator test, the test statistic is

    $$
    \begin{align}
        T_n = 2\left(l_n\left(\hat{\theta}_n^{MLE}\right) - l_n\left(\hat{\theta}_n^c\right)\right)
    \end{align}
    $$
    where $$l_n$$ is the maximum likelihood estimator. The estimator $$\hat{\theta}_n^c$$ is the constrained MLE estimator and is defined as

    $$
    \begin{align}
        \hat{\theta}_n^c = \underset{\theta \in \Theta _ 0}{\textrm{argmax}} l_n(X_1, \ldots, X_n; \theta)
    \end{align}
    $$

    Note that this estimator is always positive, and this is easily seen. Now, **Wilk's theorem** states that if $$H_0$$ is actually true, then

    $$
    \begin{align}
        T_n \xrightarrow[n \to \infty]{(d)} \chi_{d-r}^2
    \end{align}
    $$

    and hence a test at level $$\alpha$$ is $$\psi_\alpha = \mathbb{1}\left\{  T_n > q_\alpha \right\}$$, and I look up the $$\chi^2_{d-r}$$ quantile tables for hypothesis testing. 

* **Implicit Hypothesis Testing:** Consider that you are unable to test hypotheses of the form $$\theta = \theta_0$$, can you are only able to say if $$g(\theta) = 0$$ or not. You don't have access to $$\theta_0$$ but only some function of $$\theta_0$$. In this case, we can leverage the multivariate delta method and then use Wald's test to do the hypothesis testing. The formalism is as follows. Consider the estimator $$\hat{\theta}$$, we know that (for as yet unknwon $$\theta_0$$)

    $$
    \begin{align}
        \sqrt{n}(\hat{\theta} - \theta_0) \xrightarrow[n \to \infty]{(d)} \mathcal{N}(0, \Sigma(\theta))
    \end{align}
    $$

    then by the delta method we know that

    $$
    \begin{align}
        \sqrt{n}(\Gamma(\theta))^{-1/2}(g(\hat{\theta}) - g(\theta_0)) \xrightarrow[n \to \infty]{(d)} \mathcal{N}(0, I_d)
    \end{align}
    $$

    so long as the conditions for the delta method are met, and if $$\Gamma(\theta) = (\nabla g(\theta))^T \Sigma(\theta) (\nabla g(\theta))$$ is invertible. Note that in general $$g: \mathbb{R}^d \mapsto \mathbb{R}^k$$. Now, we can apply Wald's test, and if $$g(\theta_0)=0$$, we have

    $$
    \begin{align}
        n(g(\hat{\theta})^T\Gamma(\hat{\theta})(g(\hat{\theta}) \xrightarrow[n \to \infty]{(d)} \chi^2_{k}
    \end{align}
    $$

    this is where the advantage of Wald's test using $$\hat{\theta}$$ in the covariance matrix comes through, because I don't really known anythng about $$\theta_0$$; all is know is something about $$g(\theta_0)$$. In the asymptotic limit this is allowed by Slutsky. Now we can use the LHS as the estimator and design a statistical test by looking at quantile tables of the $$\chi^2_k$$ distribution.

### Lecture 12: Goodness of fit tests for discrete distributions

* Goodness of fit tests falls under the class of non-parametric hypothesis testing. We are interested in knowing if the data follows a certain distribution or not. For example, let's say I have data on race from a county, and then I have data for jurors from the county. The census data gives me a PMF for the race. Now, if I collect data on the races of the jurors and find the proportion of each race, does this fit the PMF obtained from census data? 

* We begin by introducing some formalism. Consider that I have a sample space $$E = {a_1, \ldots, a_k}$$ and statistical model $$E, \mathbf{P} _ {\mathbf{p},  p \in \Delta_k}$$. We define $$\Delta_k$$ by

    $$
    \begin{align}
        \Delta_k = \left\{  \mathbf{p} = (p_1, \ldots, p_k) \in (0,1)^K: \sum\limits_{i=1}^n p_i = 1 \right\}
    \end{align}
    $$
    If $$\mathbf{p} \in \Delta_k$$ and $$X \sim \mathbf{P} _ {\mathbf{p}}$$, then

    $$
    \begin{align}
        \mathbf{P} _ \mathbf{p}(X = a_j) = p_j
    \end{align}
    $$

    $Delta_k$ is known as the probability simplex in $$\mathbf{R}^k$$. A more compact notation for this is to say that $$Delta_k$$ is the set of all vectors $$\mathbf{p}$$ such that $$\mathbf{p} \cdot \mathbf{1} = \mathbf{p}^T \mathbf{1} = 1$$, $$p_i \geq 0$$. 

* Let $$ x_1, \ldots, X_n \sim \mathbb{P} _ \mathbf{p}$$ for some unknown $$\mathbf{p} \in \Delta _ k$$, and the $$\mathbf{p} ^ 0 \in \Delta _ k$$ be fixed. The goodness of fit test wants to test

    $$
    \begin{align}
        H_0: \mathbf{p} = \mathbf{p} ^ 0, H_1: \mathbf{p} \neq p ^ 0
    \end{align}
    $$
    with asymptotic level $$\alpha \in (0,1)$$. For example, if $$\mathbf{p} ^ 0 = (1/K, 1/K, \ldots, 1/K)$$ we are testing to see if the data follows a uniform distribution. 

* The setup above is identical to a multinomial distribution. The likelihood of the model can be written as 

    $$
    \begin{align}
        L_n(X_1, \ldots, X_n; \mathbf{p}) = p_1^{N_1}p_2^{N_2}\cdots p_K^{N_k}
    \end{align}
    $$

    where $$N_j$$ is the number of times any of the $X$ takes the value $$a_j$$. This is called the multinomial distribution with $$K$$ modalities. It model $$n$$ trials with $$K$$ possible outcomes on each trial. We can model the number of instances of each trial using a random vector $$N$$ such that $$\sum\limits_{i=1}^K N ^ {(i)} = n$$, the total number of trials, with $$n ^ {(i)} \geq 0$$. Then the pmf of the multinomial distribution is given by 

    $$
    \begin{align}
        p _ N \left(N ^ {(1)} = n ^ {(i)}, N ^ {(K)} = n ^ {(K)}\right) = \dfrac{n!}{ n ^ {(1)}! n ^ {(2)}! \cdots n ^ {(K)}! }
    \end{align}
    $$

    The likelihood function is fairly straightforward in this case, and is given above. 

* With an application of constrained optimization, we can show that $$\hat{p_j} = N_j/N$$. 

### Lecture 12: Goodness of fit tests for discrete distributions

* The $$\chi^2$$ test: this test is used to determine goodness of fit tests for the case of a discrete distribution with a pmf, such as the Bernoulli distribution, multinomial distribution, binomial distribution, etc. For the multinomial distribution, the test setup is as follows: Assume that I am testing the hypothesis 

    $$
    \begin{align}
        H_0: \mathbf{p} = \mathbf{p^0} \\
        H_0: \mathbf{p} \neq \mathbf{p^0}
    \end{align}
    $$

    Then assuming that $$H_0$$ is true, $$\sqrt{n}(\mathbf{\hat{p}} - \mathbf{p^0})$$ is asymptotically normal, and the following theorem holds

    $$
    \begin{align}
        n\sum\limits_{j=1}^{K}\dfrac{\left( \mathbf{\hat{p}_j} - \mathbf{p^0_j} \right)^2}{\mathbf{p^0_j}} \xrightarrow[n \to \infty]{(d)} \chi^2 _ {K-1}
    \end{align}
    $$

    There are some subtleties here because of the constraint that all the individual probabilities need to sum to 1. The fisher information matrix now becomes non invertible, so a lot of the previous results based on the fact that the covariance matrix is the inverse Fisher information now fails to hold. In fact, the fact, if we consider the quantity $$(\mathbf{\hat{p}} - \mathbf{p^0})^T\mathbf{1}$$, where $$\mathbf{1}$$ is the all ones vector, this dot product turns out to be precisely zero. There is no variance in this unit vector direction and hence the fisher information matrix becomes non invertible. Another way to think about this is that the asymptotic Gaussian vector lies only in $$K-1$$ dimensions because of the added dependency, and hence the degree of the $$\chi^2$$ distribution. 

* A more general version of the $$\chi^2$$ distribution for a discrete distribution. To test if some distribution $$\mathbf{P}$$ is described some family of discrete distributions $$\{\mathbf{P}\} _ {\theta \in \Theta \subset \mathbb{R^d}}$$, where $$\Theta \subset \mathbb{R}^d$$ is a $d$ - dimensional vector with support $$\{0, 1, \ldots, K \}$$ and pmf $$f_\theta$$, i.e., to test the hypothesis

    $$
    \begin{align}
        H_0: \mathbf{P} \in \{ \mathbf{P} _ \theta\} _ {\theta \in \Theta} \\
        H_1: \mathbf{P} \notin \{ \mathbf{P} _ \theta\} _ {\theta \in \Theta} \\
    \end{align}
    $$

    we define the test statistic to be

    $$
    \begin{align}
        T_n:=n\sum\limits_{j=0}^K \dfrac{\left(\frac{N_j}{n} - f_{\hat{\theta}}(j)\right)^2}{f_{\hat{\theta}}(j)} \xrightarrow[n \to \infty]{(d)} \chi_{(K+1) -d - 1}^2
    \end{align}
    $$

    Note that $$K+1$$ is the support size of $$\mathbb{P} _ {\theta \in \Theta}$$ with $$\Theta \subset \mathbb{R}^d$$, i.e., the null hypothesis $$H_0$$ holds, and if in addition some technical conditions hold, then the above test statistic allows us to define tests on the pivotal distribution. Essentially we are using proportions to define this test because in the discrete case, the MLE estimator for a parameter turns out to be the proportion of samples that was observed for each case. 

### Goodness of fit tests for continuous distributions

* The CDF of a probability distribution can be written as $$F(t) = \mathbf{P}(X \leq t) = \mathbf{E}[\mathbf{1}\{ X\leq t \}]$$. Because I see the expected value, I can now use my hammer of replacing expectations with sample means and hence I have an estimator for the CDF:

    $$
    \begin{align}
        F_n(t) = \dfrac{1}{n}\sum\limits_{i=1}^n \mathbf{1} \{ X_i \leq t \}
    \end{align}
    $$

    This quantity is called the empirical CDF or sample CDF. It just finds the proportion of outcomes that are less than or equal to $$t$$. 

* **The fundamental theorem of statistics (Glivenko-Cantelli theorem)**:

    $$
    \begin{align}
        \underset{t \in \mathbb{R}}{\sup} \lvert F_n(t) - F(t) \rvert \xrightarrow[n \to \infty]{\textrm{a.s.}} 0
    \end{align}
    $$

    The subtlety here is that Glivenko-Cantelli tells us that the convergence is uniform as opposed to the convergence being pointwise. The latter is a situation where given $$t$$, we let $$n \to \infty$$, while for uniform convergence, we pick the worst $$t$$ and then let $$n \to \infty$$. 

* The advantage of defining the empirical CDF is that because it is an average, the central limit theorem applies and hence

    $$
    \begin{align}
        \sqrt{n}(F_n(t) - F(t)) \xrightarrow[n \to \infty]{(d)}\mathcal{N}(0, F(t)(1-F(t)))
    \end{align}
    $$

    The variance is easy to calculate because each term is a Bernoulli, and the parameter of that Bernoulli is $$\mathbf{P}(X_i \leq t) = F(t)$$. 

* **Donsker's Theorem**

    The theorem states that 

    $$
    \begin{align}
            \sqrt{n} \underset{t \in \mathbb{R}}{\sup}{\lvert F_n(t) - F(t)\rvert} \xrightarrow[n \to \infty]{(d)} \underset{0 \leq t \leq 1}{\sup}\lvert \mathbb{B}(t) \rvert
    \end{align}
    $$

    Here, the quantity $$\mathbb{B}(t)$$ is called the Brownian bridge. This is a distribution that simulates at a random walk between 0 and 1 under the constraint that the two ends are pinned at 0. Each step is itself a standard Gaussian. The Kolmogorov-Smirnov test is a natural extension of this theorem because the Brownian bridge is a pivotal distribution.

* **Kolmogorov-Smirnov test**

    Assume that we want to test the null hypothesis $$H_0: F = F^0$$ against the alternative hypothesis $$H_1: F \neq F^0$$. Define the test statistic based on the theorem above 

    $$
    \begin{align}
        T_n := \sqrt{n} \underset{t \in \mathbb{R}}{\sup}{\lvert F_n(t) - F^0(t)\rvert}
    \end{align}
    $$

    then we know from Donsker's theorem that 

    $$
    \begin{align}
        T_n \xrightarrow[n \to \infty]{(d)} Z
    \end{align}
    $$

    where $$Z$$ has a known distribution, i.e., the supremum of the absolute value of a Brownian bridge. Now, for the KS test with asymptotic level $$\alpha$$, we define 

    $$
    \begin{align}
        \delta _ \alpha ^ {KS} = \mathbf{1} \{ T_n > q _ {\alpha} \}
    \end{align}
    $$

    where $$q_\alpha$$ is the $$1-\alpha$$ quantile of the supremum of the absolute value of a Brownian bridge. The p-value can then be calculated as $$\mathbf{P}(Z > T_n \vert T_n)$$, where $$Z$$ is a random variable described by the supremum of the absolute value of a Brownian bridge. 

* How do I actually calculate this supremum in practice? Let us say we have a set of observations $$X_1, \ldots, X_n$$. We reason that given the step wise nature of the empirical CDF, we should see the largest discrepancy between $$F_n(t)$$ and $$F(t)$$ at the points of observation. Therefore, the first thing we do is to order all the observation points in ascending order. Such ordered statistics are notated by $$X _ {(1)}, \ldots, X _ {(n)}$$. We can then simplify the calculation of the supremum as 

    $$
    \begin{align}
        T_n = \sqrt{n}\underset{i = 1, \ldots, n}{\max} \left( \max\left( \left\lvert \dfrac{i-1}{n} - F^0(X _ {(i)}) \right\rvert, \left\lvert \dfrac{i}{n} - F^0(X _ {(i)}) \right\rvert \right) \right)
    \end{align}
    $$

    where there are a total of $$n$$ points. Note that we have used the fact that the value of the empirical CDF at $$X _ {(i)}$$ just counts the proportion of points less than or equal to $$X _ {(i)}$$. 

* There still remains the issue of how I actually go about calculating the supremum of the absolute value of the Brownian bridge. There is something remarkable that happens here that allows us to calculate this distribution in a non-asymptotic manner. Let $$X _ 1, \ldots, X _ n$$ de distributed with CDF $$F$$. Define a new random variable $$Y _i = F(X _ i)$$. Then the CDF of $$Y _ i$$ is given by

    $$
    \begin{align}
        F _ {Y _ i} (t) & = \mathbf{P}(Y _ i \leq t) \\
                        & = \mathbf{P}(F(X _ i) \leq t) \\
                        & = \mathbf{P}(X _ i \leq F ^ {-1} (t))\\
                        & = F(F ^ {-1} (t)) = t
    \end{align}
    $$

    Note that $$t \in [0,1]$$ because $$Y _ i$$ is a CDF. This means that the CDF of $$Y _ i$$ is the same as a uniform random variable and $$Y _ i = F(X_i) \sim \textrm{Unif}([0,1])$$. Therefore, no matter what the actual CDF of $$X _ i$$ is, I can rewrite in non asymptotic form that 

    $$
    \begin{align}
        T_n & := \sqrt{n} \underset{t \in \mathbb{R}}{\sup}{\lvert F_n(t) - F^0(t)\rvert} \\
            & = \sqrt{n}\underset{0 \leq x \leq 1}{\sup} \left\lvert G_n(x) - x \right\rvert \\
            & = \sqrt{n}\underset{0 \leq x \leq 1}{\sup} \left\lvert \dfrac{1}{n} \sum\limits_{i=1}^n \mathbf{1}\{ X_i \leq x \} - x \right\rvert
    \end{align}
    $$

    where $$G_n(x)$$ is the empirical CDF of a uniform distribution. Therefore, to actually calculate the supremum of the absolute value of the Brownian bridge, I first discretize my domain of $$ 0 \leq x \leq 1$$ into a fine mesh, and then for each one of those $x_i$, I simulate $$n$$ uniform random variables and find $$T_n ^ l $$. I then simulate $$T_n$$ like this $$M$$ times, where $$M$$ is very large. I then know that $$T_n ^ l \to T_n$$ as $$M$$ grows large, and I use this value as the supremum of the Brownian bridge. 

    In any case, there are tables that I can look up to find quantiles of the Kolmogorov-Smirnov test (i.e. for the supremum of the absolute value of a Brownian bridge), so I don't have to calculate this thing each time, but this is just a useful technique to generate samples of any given distribution starting from being able to generate samples for a uniform distribution.

    Here is some python code below that generates the distribution for the supremum of the absolute value of the Brownian bridge using a uniform distribution:

    ```python
    import numpy as np
    from scipy import stats
    import matplotlib.pyplot as plt

    # Basic constants
    x = np.linspace(0, 1, 1000)
    n = 1000

    def statistic_calculator(x):
        """
        Given x, this function generates n iid copies of uniform 
        random variables, and then calculates the value in the abs
         signs in equation 1.
        input: x
        returns: value of quantity in absolute value in equation 1
        """
        # generate n rvs
        nrvs = stats.uniform.rvs(loc=0, scale=1, size=n)
        emp_cdf = np.sum((nrvs < x)*1) / n
        return abs(emp_cdf - x)
    
    # vectorize the function
    vec_statistic_calculator = np.vectorize(statistic_calculator)

    def tlcalculator():
        """
        Does the optimization to find the supremum over the domain 
        0 <= x <= 1
        """
        return np.max(vec_statistic_calculator(x))
    
    # Do this a whole bunch of times for large M to find the 
    # distribution
    M = 22000
    m_all = np.array([tlcalculator() for i in range(M)])
    plt.hist(m_all, bins=100);
    ```

* **Kolmogorov-Lilliefors test**: The Kolmogorov-Smirnov test is useful in the case that I am given a certain Gaussian distribution of known parameters, and I would like to test if my data agrees with the hypothesis that it is distributed according to this Gaussian. I can the for example use the student t-test to proceed further if the goodness of fit test determines this to be true (i.e. I fail to reject). However, what about the case where I do not know what the parameters of the Gaussian distribution is? It would be natural to assume that I can use estimators

    $$
    \begin{align}
        \hat{\mu} & = \bar{X} _ n \\
        \hat{\sigma^2} & = S _ n
    \end{align}
    $$

    But by doing this, I am already making my data look more Gaussian because I am using the data to check something about the data. I am estimating the unknown parameters based on the data and this is dangerous. In this case, Donsker's theorem does not apply! Instead I have to correct for the fact that I am less likely to reject the hypothesis by decreasing the values of the quantiles $q _ \alpha$ (compared to the Kolmogorov-Smirnov test). These tables exist. 

    $$
    \begin{align}
        \sqrt{n}\underset{t \in \mathbb{R}}{\sup} \left\lvert F _ n(t) - \Phi _ {\hat{\mu}, \hat{\sigma^2}}(t) \right\rvert \xrightarrow[n \to \infty]{(d)} Z
    \end{align}
    $$

    where $$Z$$ is a random variable whose distribution is given by the Kolmogorov-Lilliefors tables. 

* Quantile-Quantile plots are an informal but visual way to test goodness of fits. It's often hard to draw two CDFs and then check to see if they look close. So we flip it by plotting quantiles (i.e. the inverse of the CDF) against each other. If the two distributions being compared are exactly the same, we would expect a line that is parallel to $$y=x$$. In reality, we will see deviations from this line to different extents, either on the left axis, or on the right axis or both. More formally, we are plotting the points

    $$
    \begin{align}
        \left( F ^ {-1} (i/n), F _ n ^{-1}(i/n) \right)
    \end{align}
    $$

    The inverse CDF is somewhat confusing to think about. In essence, what we need to find for the empirical CDFis the $$i$$-th ordered observation $$X _ {(i)}$$. For the continuous CDF, we are interested in finding that value of $$x _ i$$ such that $$F(x _ i) = i/n$$. One can intuit how this quantifies the distance between the empirical CDF and the true CDF. The Q-Q plot needs to be interpreted very carefully. For example, we know that the student distribution has heavier tails compared to the Gaussian distribution. So we would expect the quantiles for the student t on the right to be larger (i.e. more to the right) than that of the Gaussian distribution. Similarly, the quantiles on the left would be more to the left than those of the Gaussian. However, if I don't have enough data, I would be very tempted to conclude that the student distribution is in fact Gaussian because I wouldn't have enough data to see these small deviations at the tails. Try and think about what you would expect the deviations on the QQ plots to be for some standard distributions such  as uniform, exponential, etc. compared to the standard normal.

## Unit 5: Bayesian Statistics

### Lecture 17: Introduction to Bayesian Statistics

This unit doesn't have that much new in terms of Bayesian thinking over the probability class so just see those notes. Below are just some brief notes on new material. 

* Remind yourself of frequentist and Bayesian approaches based on the notes in the probability class.

* When we have a Bernoulli parameter to estimate, we often use the Beta distribution (with some chosen values for $$a, b$$) as a flexible example of the prior.

* A prior is called a conjugate prior if the posterior probability falls in the same family as the prior itself. For example, a Gaussian prior and data that is distributed as a Gaussian, or a Gamma prior and data that is distributed through a Gamma distribution, etc.

* An improper prior is one that does not integrate to 1. For example, assuming a uniform prior when the data is distributed as a Gaussian. It of course doesn't matter because depending on the data, the posterior can still be proper and well-behaved. 

* **Jeffrey's prior**: This is an important new thing learned in this class. The idea is that given the nature of the distribution of the observation $$X_1, \ldots, X_n$$, we can find the Fisher information matrix $$I(\theta)$$ of the probability model (assuming that it exists). Jeffrey's prior is then defined as 

$$
\begin{align}
    \pi_J(\theta) \propto \sqrt{ \det{I(\theta)} }
\end{align}
$$

* The idea behind the Jeffrey's prior is quite interesting. We know that the Fisher information gives us some information about how much we know about the distribution i.e. the variance of the distribution is the inverse of the Fisher information. The Fisher information gives us the curvature of the likelihood function (can you reason why?). So by making the Jeffrey's prior proportional to the Fisher information, we, are giving weight to those points where we know a lot about the distribution (lots of information and low variance)

## Unit 6: Linear Regression

## Unit 7: Generalized Linear Models

### Lecture 21: Introduction, Exponential Families
