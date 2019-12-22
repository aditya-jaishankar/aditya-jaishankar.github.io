---
title: "Probability: The Science and Uncertainty of Data (6.431x)"
categories:
toc: true
layout: single
classes: wide
permalink: /coursenotes/probability/
author_profile: true
toc: true
date: September 2018
read_time: true
---

$\require{\cancel}$

## Unit 1: Probability models and axioms

### Lecture 1: Probability models and axioms


* The basic concept in probability is that of a sample space.

* Probability laws help calculate probabilities of a particular event in the sample space from occurring. These laws have to respect axioms, for example, probabilities cannot be negative; the axioms are few, but powerful.

* There are three chief requirements for a sample space $\Omega$:

    0. That the elements of the set are mutually exclusive i.e. only any one event can occur simultaneously at a time,
    0. The elements of the set need to be collectively exhaustive i.e. the set captures all possible outcomes
    0. Must be at the right granularity i.e. the probability model must be able to leave out unnecessary details, for example whether it is raining outside.

    (1) and (2) together means that one and only one outcome in the sample space **must** occur. 

* Sample spaces can be discrete, finite, infinite, continuous, etc. (This listing is not collectively exhaustive)

* For easily countable number of outcomes, you can either draw up an outcome table or a outcome tree to enlist all possible outcomes.

* For continuous sample sets, the probability of any one event occurring is 0, because the list of possible outcomes is infinite. All we can talk about in this case is the probability of a subset of the sample space. For example, probability of throwing a dart at the bulls-eye is zero, but the probability of it ending up in some sub-disk of the board is finite.

* Here are some axioms of probabilities (Consider sample set $\Omega$ and an event $A$):

    $$
    \begin{align}
        \mathbf{P}(A) & \geq 0 \\
        \mathbf{P}(\Omega) &= 1 \\
        \mathbf{P}(A \cup B) & = \mathbf{P}(A) + \mathbf{P}(B)
    \end{align}
    $$

    if $\mathbf{P}(A \cap B) = \phi$, i.e., if $A$ and $B$ are disjoint (mutually exclusive) events. 

* Definition: A subset of a sample space is called an \textbf{event}. 

* A generalization of summation axiom to a set of discrete events $\{s_1, s_2, ..., s_n\}$ is that $\mathbf{P}(\{s_1, s_2, ..., s_n\}) = \sum^i\mathbf{P}(\{s_i\}) = \sum^i\mathbf{P}(s_i)$. This can be proved like the continuous case, but considering subsets consisting of one element, event $s_i$. By construction, these subsets are disjoint. 

* When proving theorems on sets, subsets and probabilities, use Venn diagrams. They are your friend. 

* Some more theorems: if $A \subset B$,

    $$
    \begin{align}
        & \mathbf{P}(A \cup B)  = \mathbf{P}(A) + \mathbf{P}(B) - \mathbf{P}(A \cap B) \\
        & \textrm{Union bound:} \mathbf{P}(A \cup B)  \leq \mathbf{P}(A) + \mathbf{P}(B)
    \end{align}
    $$

* Discrete uniform law: if $\Omega$ has $n$ discrete outcomes, each equally likely (so each outcome has probability $1/n$) and $A \subset \Omega$ has $k$ elements, then $\mathbf{P}(A) = k \cdot 1/n = k/n$. 

* Probability laws are somewhat arbitrary because they are a model. We chose a model of what best captures the situation. 

* Uniform probability law: For continuous sample spaces, the probability of a subset $A$ of $\Omega$ (i.e. $A \subset \Omega$), is the ratio of the areas of $A$ and $\Omega$. 

* There are four steps to the calculations of probabilities
    0. Identify a sample space
    0. Identify probability laws (counting, uniform probability law, etc.)
    0. Identify an event of interest
    0. Calculate
    

* When calculating probabilities, drawing pictures, trees, tables, enumeration, etc.\ is highly recommended. Visual cues help a lot. 

* The probability law has to be able to calculate probabilities for every possible outcome i.e. any subset $A$ of $\Omega$. 

* **Countable additivity theorem:** For any _sequence_ of _disjoint_ events $A_i$,

    $$
    \begin{align}
    \mathbf{P}(A_1 \cup A_2 \cup \cdots) = \sum_{i = 1}^\infty \mathbf{P}(A_i)
    \end{align}
    $$

* The word _sequence_ is very important here. The word sequence can be interpreted as discrete events. For example, if the sample space is continuous, there is no way to arrange the elements in a sequence, and this relationship does not hold. For example, the real line cannot be arranged in a sequence.

* I cannot assign uniform probabilities to an infinite sequence of events in the sample space.

### Lecture 2:  Math background: Sets; sequences, limits, and series; (un)countable sets.

* There are fundamental differences in the approaches used for discrete sample spaces and continuous sample spaces. 

* A **set** is a collection of **distinct** elements.

* Notation: 
    * $\{a, b, c, d\}$ denotes a finite set where we can enumerate elements.
    * For uncountable sets, $\{x \in \mathbb{R}: \cos(x) > 1/2\}$ means set of all real numbers $x$ such that $\cos(x) > 1/2$.
    * $\bigcup\limits_n S_n$ denotes unions of all sets $S_i$. Therefore $x \in \bigcup\limits_n S_n \iff x \in S_i$ for some $i$ in $1, 2, ..., n$. Similarly, intersections can be defined.

* **De Morgan's laws:** These can be proved with some basic logical arguments. 

    $$
        \begin{align}
            \left(\bigcup\limits_n S_n\right)^c & = \bigcap\limits_n S_n^c\\
            \left(\bigcap\limits_n S_n\right)^c & = \bigcup\limits_n S_n^c
        \end{align}
    $$


* A **sequence** {$a_i$} can be thought of as a function that assigns a value to every natural number, i.e., $f:\mathbb{N} \to a_i, a_i \in S$. 


* What we care about a lot is **convergent sequences**, where $\lim\limits_{i \to \infty} a_i = a$. The formal definition is that {$a_i$} converges to $a$ if for any arbitrary $\varepsilon > 0, \varepsilon \in \mathbb{R}$ there exists $i > i_0$ such that $\lvert a_i - a\rvert < \varepsilon$ for some large enough $i_0 \in \mathbb{N}$.



    
* In an infinite series $\sum^\infty a_i$, the series is well defined if $a_i \geq 0$; the limit is either finite or infinite. That is, if the sequence of partial sums is monotonic as $n$ is made larger the limit exists (could be $\infty$). If the sequence of partial sums is non-monotonic i.e. if $a_i$ can be negative, then it is possible the sum of the series exists, but also that if we rearrange the order of performing the sum, we may get a different limit. These problems can be avoided by considering $S = \sum^\infty \lvert a_i\rvert$. If $S < \infty$ for *some* particular order of arranging the terms, then the original sequence of partial sums is guaranteed to converge to a finite limit, regardless of the particular order of adding the terms. This condition also applies to double series $\sum\limits_{i, j}^\infty a_{ij}$. If $\sum \lvert a_{ij}\rvert < \infty$ for *some* particular ordering of the terms, then the series (i.e. the sequence of partial sums) converges to a finite value, and order of summation of terms does not matter. *The lecture includes a beautiful graphical example of adding row-wise and column-wise to demonstrate this point.*

* **Countable** and **Uncountable** sets: if I can assign a positive integer index to every element in the sample space, then the set is known as countable. In other words, if I can arrange the elements of the set in a sequence with positive integer indices. For example, the positive integers, all integers, rational numbers, etc. However, any interval on the real line is uncountable.  Cantor's diagonalization theorem is a beautiful way to prove this.

#### Solved problem set: Bonferroni's inequality
* For any $n$ events $A_1, A_2, ..., A_n$,

    $$
        \begin{align}
            \mathbf{P}(A_1 \cap A_2 \cap \cdots \cap A_n) \geq \mathbf{P}(A_1) + \mathbf{P}(A_2) + \cdots \mathbf{P}(A_n) - (n-1)
        \end{align}
    $$
    
## Unit 2: Conditioning and Independence

### Lecture 2: Conditioning and Bayes' Rule

* The conditional probability of event $B$ occurring, given that event $A$ has occured is given by

    $$
    \begin{align}
    \mathbf{P}(B|A) = \dfrac{\mathbf{P}(B \cap A)}{\mathbf{P}(A)}
    \end{align}
    $$

* This is a definition - there is no question about whether it is correct or not. There is a useful intuition trick based on redefining the sample space i.e. when talking about $\mathbf{P}(B\lvert A)$, because we are given that $A$ definitely occured, all the elements in the sample space in $A^c$ are automatically eliminated. $A$ becomes the new sample space. 



* This is a definition - there is no question about whether it is correct or not. There is a useful intuition trick based on redefining the sample space i.e. when talking about $\mathbf{P}(B\lvert A)$, because we are given that $A$ definitely occured, all the elements in the sample space in $A^c$ are automatically eliminated. $A$ becomes the new sample space. 

* It can be shown that conditional probabilities obey the axioms of probabilities. This is powerful because any results derived for regular probabilities also apply to conditional probabilities. 

* In general, it can be shown that for $n$ events $A_1, A_2, \cdots, A_n$, 

    $$
        \begin{align}
        \mathbf{P}(A_1 \cap A_2 \cap \cdots \cap A_n) = \mathbf{P}(A_1) \prod\limits_{k=2}^n \mathbf{P}(A_k|A_1 \cap A_2 \cap \cdots \cap A_{k-1})
        \end{align}
    $$

* This derives straight from a repeated application of 

$$
\begin{align}
\mathbf{P}(A_1 \cap A_2 \cap \cdots \cap A_n) & = \mathbf{P}((A_1 \cap A_2 \cap \cdots \cap A_{n-1}) \cap A_n)\\
& = \mathbf{P}(A_1 \cap A_2 \cap \cdots \cap A_{n-1})\cdot\mathbf{P}(A_n | A_1 \cap A_2 \cap \cdots \cap A_{n-1})
\end{align}
$$

* In words, the probability of $A_1 \cap A_2 \cap \cdots \cap A_n$ occurring is the probability of $A_1 \cap A_2 \cap \cdots \cap A_{n-1}$ occurring times the probability of $A_n$ occurring given that $A_1 \cap A_2 \cap \cdots \cap A_{n-1}$ occurred. 

*  **Total Probability theorem:** If $A_1, A_2, \cdots, A_n$ are disjoint events, then

    $$
    \begin{align}
    \mathbf{P}(B) = \sum\limits_{k=1}^n \mathbf{P}(A_i)\cdot \mathbf{P}(B|A_i)
    \end{align}
    $$

    In other words, the total probability that an event $B$ occurs is the weighted sum of the probabilities $B$ occurs under all disjoint scenarios, weighted by the probability of each scenario.  

* **Bayes Theorem:** Consider that we have some set of disjoint events $A_i$, each with probability $\mathbf{P}(A_i)$. These are our **prior beliefs**. Let $B$ be an event which could occur under the possible scenarios $A_i$, all of which are disjoint. Bayes' rule essentially allows for updating our prior beliefs given open the observation that $B$ occurred. The rule states that

    $$
    \begin{align}
    \mathbf{P}(A_i|B) = \dfrac{\mathbf{P}(A_i) \cdot \mathbf{P}(B|A_i)}{\sum\limits_{i=1}^\infty \mathbf{P}(A_i) \cdot \mathbf{P}(B|A_i)}
    \end{align}
    $$

    Note that the infinite sum only applies in the case the events $A_i$ are a \textbf{sequence} of disjoint events (refer countable additivity theorem). 

* The term inference, which we look at in a lot more detail later in the course, is the process of updating prior beliefs given the outcome of an event under the scenario of the belief. 

### Lecture 3: Independence

* Two events are considered independent if the occurrence of one event doesn't change the probability of the other event. One vent doesn't change our beliefs about the other event. 

* More formally, two events $A$ and $B$ are independent if $\mathbf{P}(A \cap B) = \mathbf{P}(A) \cdot \mathbf{P}(B)$. 

* Some intuition here: Independence is completely different from two events being disjoint. In fact, if two events are disjoint, the occurrence of one event immediately means that the other event did not occur. So disjoint events are like Siamese twins, so to speak. 

* Independence usually arises when the events occur due to some non-interacting physical processes. For example, whether I toss a heads and whether it will rain on New Year's Day. 

* If $A$ and $B$ are independent events, then $A^c$ and $B^c$ are also independent, and so are $A$ and $B^c$. Think about this intuitively. 

* **Conditional Independence:** Two events $A$ and $B$ are conditionally independent if

    $$
    \begin{align}
    \mathbf{P}(A \cap B)|C) = \mathbf{P}(A|C) \cdot \mathbf{P}(B|C)
    \end{align}
    $$

* Note that if two events $A$ and $B$ are independent, it does not automatically imply that they are conditionally independent. Can you think of a Venn diagram example as to why this is true?

* Independence of multiple events: Events $A_i$ are independent if $\mathbf{P}(A_1 \cap A_2 \cap \cdots \cap A_n) = \mathbf{P}(A_i) \mathbf{P}(A_j) \cdots \mathbf{P}(A_m)$ for any choice and any number of $i, j$, and $m$. 

* For a collection of events, independence is a very difference issue from pairwise independence. Pairwise independence for all pairs does not imply that all events are independent.


## Unit 3: Counting

### Lecture 4: Counting
* Counting techniques can be used whenever the elements of the sample space are equally likely i.e. the probability of an event is proportional to the number of events in that sample space. 

* The binomial coefficient is defined as

    $$
    \begin{align}
    \binom{n}{k}
    \end{align}
    $$

* This is the same as $^n C_k$ and is very useful is calculating binomial probabilities. For example, if I have a biased coin that gives heads with probability $p$, what is the probability that in $n$ tosses I will get $k$ heads.  

* The binomial coefficient can be expanded as:

    $$
    \begin{align}
    (a + b)^n = \sum\limits_{k=0}^n \binom{n}{k} a^k b^{n-k}
    \end{align}
    $$

* *Keep in mind the physical interpretation of the RHS of the above equation:* The quantity $\binom{n}{k}p^k (1-p)^{n-k}$ is the probability that an unfair coin with probability $p$ of getting a head will give $k$ heads when tossed $n$ times; these heads can be mixed and arranged in $\binom{n}{k}$ ways. Also, the sum in the above equation (with $a = p$ and $b = 1-p$) covers the entire sample space, i.e., the probability of getting 1 head or 2 heads or, $\cdots$, $n$ heads, and therefore that sum is one. Very useful physical interpretation of binomial probabilities.   

## Unit 4: Discrete Random Variables

### Lecture 5: Probability mass functions and expectations

* A **random variable** is a variable whose value depends on the outcome of a probabilistic experiment. A **discrete random variable** is a random variable whose value is in a finite or countably infinite set. The number of possible values a discrete random variable can take is finite or countably infinite. 

* A random variable can be thought of as a function that takes a particular element in the sample space $\Omega$ and maps it to a particular value (in the case of a discrete random variable, the value is in a finite set or a countably infinite set). 

* **Notation:** Random variables are denoted by a capital letter $X$ as opposed to the value of a particular outcome of the random variable, which is denoted in lower-case letters $x$. 

* We can define a random variable that is a function of random variable i.e. $Z = X + Y$ which produces the value $z$ when the value of $X$ is $x$ and the value of $Y$ is $y$. 

* A key concept is that of a **\textbf{**probability mass function (PMF)** or the **probability distribution** which determines the probability that a discrete random variable $X$ takes value $x$. The notation is $p_X(x)$ or $\mathbf{P}(X=x)$. Some properties: $p_X(x) \geq 0$ and $\sum_i p_X(x_i)=1$ where $x_i$ captures every element of the sample space (these are all disjoint because the sample space here is considered to be discrete. For example, think about two successive rolls of a die. Let the value of the first roll is described by the discrete random variable $X$ and the second be $Y$. Define $Z = X+Y$. Then $p_Z(2) = 1/16$, $p_Z(3) = 2/16$ (the outcome that $X=1$ and $Y=2$ or $X=2$ and $Y=1$), etc. 

* The formal definition of a probability mass function is 

    $$
    \begin{align}
    p_X(x) = \mathbf{P}(\{\omega \in \Omega : X(\omega)=x\})
    \end{align}
    $$

* In words, the probability distribution or the probability mass function is described as the probability of the event containing those elements $\omega \in \Omega$ such that the value of the discrete random variable $X(\omega) = x$. Find all the event consisting of the elements $\omega$ for which the value of the discrete random variable $X$ at those $\omega$ is $x$.  

* **Bernoulli Random Variables**: They only take two values 0 and 1 with probability $1-p$ and $p$ respectively. i.e the PMF can be written as

    $$
    \begin{align}
    p_X(x) & =  p \textrm{ if } x=1,\\
    & = 1-p \textrm{ if } x=0
    \end{align}
    $$

* Bernoulli Random Variables can be linked to _indicator random variables_ $I_A$. These random variables tell us whether an event $A$ occurred or its complement $A^c$ occurred i.e. $A$ did not occur. $I_A =1$ if $A$ occurred and 0 if $A^c$ occurred. 

    $$
    \begin{align}
    p_{I_A}(1) = \mathbf{P}(I_A = 1) = \mathbf{P}(A)
    \end{align}
    $$

    Indicator random variables become very useful later when trying to break down a complex problem into smaller steps.

* **Discrete uniform random variable:** Parameters $a, b$. The random variable $X(\omega) = \omega$ and $\omega \in \\{a,b\\}$ where $a$ and $b$ are integers. Therefore there are $b - a +1$ integers in the range. The PMF for this random variable is

    $$
    \begin{align}
    p_X(x) = \frac{1}{b-a +1}
    \end{align}
    $$

* **Binomial random variable:** Parameters $n ,p$. This random variables describes the number of successes in an experiment consisting of $n$ trials, all identical, where a 'success' occurs with probability $p$ and a failure with probability $1-p$. For example, tosses of a biased coin with probability $p$ of getting a head. The random variable that describes the number of heads is a binomial random variable. For a binomial random variable $X$, 

    $$
    \begin{align}
    p_X(k) = \binom{n}{k}p^k(1-p)^{n-k}
    \end{align}
    $$

* **Geometric random variable:** Parameter $p$. Let the experiment be a sequence of coin tosses and the random variable $X$ to be the first occurrence of a head. Then $p_X(k)$ describes the probability that the first head is achieved on the $k$-th toss. Therefore we have (and think about why this is the result) $p_X(k) = (1-p)^{k-1} p$

* The **expected value** of a random variable is defined as

    $$
    \begin{align}
    \mathbf{E}[X] = \sum\limits_x x\cdot p_X(x)
    \end{align}
    $$

    The physical interpretation of the expected value is the average value of a random variable one would expect after a very large number of experiments.

* Some expected values of common random variables:

    $$
    \begin{align}
    \textrm{Bernoulli random variable:} & \mathbf{E}[X] = p \\
    \textrm{Uniform random variable:} & \mathbf{E}[X] = n/2
    \end{align}
    $$

* Expected value rule for calculating the expected value of a function of a random variable (**Expected value theorem**):
Assume you have a random variable $X$ with PMF $p_X(x)$ and another random variable $Y = g(X)$. The expected value of $Y$ is given by

    $$
    \begin{align}
    \mathbf{E}[Y] = \sum\limits_x g(x)p_X(x)
    \end{align}
    $$

    Think about proof.

* Expected values are linear i.e. $\mathbf{E}[aX + b] = a\mathbf{E}[X] + b$


### Lecture 6: Variance, conditioning on an event; multiple random variables

* The variance of a random variable $X$, denoted by $\textrm{var}(X)$ or $\mathbf{E}[(x-\mu)^2]$, where $\mu$ is the expected value of $X$ gives you a sense for the spread of the distribution. i.e. it captures the uncertainty of the random variable. If $X$ is very uncertain, or random, the variance is large. If $X$ is deterministic, the variance is 0. A more dimensionally consistent quantity is the standard deviation $\sigma = \sqrt{\textrm{var}(X)}$ 

* Useful relationship: $\textrm{var}(aX +b) = a^2 \textrm{var}(X)$. *Think about proof.*

* $\textrm{var}(X) = \mathbf{E}[X^2] - (\mathbf{E}[X])^2$. $\mathbf{E}[X^2]$ can be calculated from an application of the expected value theorem with $g(x) = x^2$.  

* Variances of common random variables: 

    $$
    \begin{align}
    \textrm{Bernoulli random variable:} &\textrm{var}(X) = p(1-p) \\
    \textrm{Uniform random variable: } & \textrm{var}(X) = \dfrac{1}{12}n(n+2) = \dfrac{1}{12}(b-a)(b-a+2)\textrm{ if $n = b-a$}.  
    \end{align}
    $$

**Conditioning on random variables**

* Every probability theorem or model has a conditioning counterpart. This applies to random variables as well. Therefore I can define a conditional PMF analogous to a regular PMF. 

* Notation: $p_{X\vert A}(x) = \mathbf{P}(X = x\lvert A)$ which is interpreted as the probability of the random variable $X$ taking the value $x$ given that event $A$ occurred. 

* All other results remain similar, for example:

    $$
    \begin{align}
    \mathbf{E}[X|A] & = \sum\limits_x x p_{X|A}(x) \\
    \textrm{var}(X|A) & = \sum\limits_x (x-\mu)^2 p_{X|A}(x) = \mathbf{E}[X^2|A] - (\mathbf{E}[X|A])^2
    \end{align}
    $$

    where $\mu$ is the expected value.

* **Total expectation theorem:**

    This theorem allows us to divide and conquer the calculation of expected values. Consider a sample space $\Omega$ that has been into $n$ disjoint events $A_1, A_2, \cdots, A_n$. Consider a random variable $X$ and an event $B$ which is the event that $X=x$. First, from the total probability theorem, 

    $$
    \begin{align}
    \mathbf{P}(B) = \sum\limits_i \mathbf{P}(B|A_i)\cdot \mathbf{P}(A_i)
    \end{align}
    $$

    Now $\mathbf{P}(B\vert A_i) = \mathbf{P}({X=x}\vert A_i) = p_{X\vert A_i}(x)$ and therefore the above equation can be written as

    $$
    \begin{align}
    p_X(x) = \sum\limits_i p_{X|A_i}(x)\cdot \mathbf{P}(A_i)
    \end{align}
    $$

    Multiplying both sides by $x$ and summing over all $x$,

    $$
    \begin{align}
    \mathbf{E}[X] = \sum\limits_i \mathbf{E}[X|A_i] \cdot \mathbf{P}(A_i)
    \end{align}
    $$


* Geometric random variables display *memorylessness* and the following result holds:

    $$
    \begin{align}
    \mathbf{E}[X] = \mathbf{E}[X-n|X>n]
    \end{align}
    $$

    This kind of approach helps us break down the problem in a clever way and use divide and conquer to calculate expected values. For example, to calculate the expected value of a geometric random variable,

    $$
    \begin{align}
    \mathbf{E}[X] = \sum\limits_{k=1}^n k (1-p)^{k-1} p
    \end{align}
    $$

    With an application of the total expectation theorem, we have

    $$
    \begin{align}
    \mathbf{E}[X] & = 1+ \mathbf{E}[X-1]\textrm{ (from linearity of expectations)}\\
    & = 1 +p\cdot \mathbf{E}[X-1|X=1] + (1-p) \cdot \mathbf{E}[X-1|X>1]
    \end{align}  
    $$

    The above stems from the total expectation theorem. The event is split into the first toss being a head, and everything else. Therefore,

    $$
    \begin{align}
    \mathbf{E}[X] & = 1 + 0 + (1-p) \mathbf{E}[X]\\
    & = \dfrac{1}{p}
    \end{align}
    $$

* The fact that $\mathbf{E}[X] = \mathbf{E}[X-n \lvert x>n]$ are the same is somewhat intuitive. The distributions are the same; this can be seen by replacing $Y = X-4$. The condition $X>4$ is the condition $Y>0$ which means $Y = 1, 2, \cdots$, and this is exactly the geometric random variable. 

* Joint PMFs are rather intuitive when single variable PMFs are understood. The following results hold:

    $$
    \begin{align}
    & p_{X,Y}(x,y)  = \mathbf{P}(X = x\textrm{ and } Y = y) \\
    & \sum\limits_x \sum\limits_y p_{X,Y}(x,y) = 1 \\
    & p_X(x) = \sum\limits_y p_{X,Y}(x,y) \textrm{ (called marginal PMF)} \\
    \end{align}
    $$

    If $Z = g(x,y)$,

    $$
    \begin{align}
    p_Z(z) = \!\!\!\!\!\!\!\!\!\!\!\!\sum\limits_{(x,y): g(x,y)=z}\!\!\!\!\!\!\!\!\!\! p_{X,Y}(x,y)
    \end{align}
    $$

    Moreover, there is also a expected value rule:

    $$
    \begin{align}
    \mathbf{E}[g(x,y)] = \sum\limits_{(x,y)}g(x,y)p_{X,Y}(x,y)
    \end{align}
    $$

* Expected value of a Bernoulli random variable with parameters $n,p$ is $\mathbf{E}[X] = np$

### Lecture 7: Conditioning on a r.v.; Independence of r.v.'s

* The notation for a conditional PMF is $p_{X\vert Y}(x\vert y)$. Intuitively, this is the same as $\mathbf{P}(X=x \vert Y=y)$ i.e. the probability that $X=x$ given that $Y=y$. We can define this in a similar fashion to regular conditional probabilities i.e.

    $$
    \begin{align}
    p_{X|Y}(x|y) = \dfrac{p_{X,Y}(x,y)}{p_Y(y)}
    \end{align}
    $$

    for some particular $y$. There are a whole family of conditional probabilities corresponding to every single value that $y$ can take. 

* Similar to the product rule for conditional probabilities, 

    $$
    \begin{align}
    p_{X,Y}(x,y) = p_Y(y)\cdot p_{X|Y}(x|y)
    \end{align}
    $$

* Total probability theorem for conditional PMFs:

    $$
    \begin{align}
    p_X(x) = \sum\limits_y p_Y(y) p_{X|Y}(x|y)
    \end{align}
    $$

* The expected value of a conditional PMF:

    $$
    \begin{align}
    \mathbf{E}[X|Y=y] = \sum\limits_x x p_{X|Y}(x|y)
    \end{align}
    $$

* Expected value rule for conditional PMFs:

    $$
    \begin{align}
    \mathbf{E}[g(X)|Y] = \sum\limits_x g(x) p_{X|Y}(x|y)
    \end{align}
    $$

* Total expectation theorem for conditional PMFs:

    $$
    \begin{align}
    \mathbf{E}[X] = \sum\limits_y p_Y(y) \mathbf{E}[X|Y=y]
    \end{align}
    $$

* Conditional independence of joint PMFs: To calculate this, pick an $(x, y)$ and check if \mbox{$p_{X,Y}(x,y) = p_X(x) p_Y(y)$} and then do this for every $(x, y)$. 

* Interesting property of expectations: \textit{if random variables $X$ and $Y$ are independent}, then

    $$
    \begin{align}
    \mathbf{E}[g(X)h(Y)] = \mathbf{E}[g(X)]\mathbf{E}[h(Y)]
    \end{align}
    $$

    where $g$ and $f$ are functions. 

* Interesting property of expectations: \textit{if random variables $X$ and $Y$ are independent}, then

    $$
    \begin{align}
    \textrm{var}(X+Y) = \textrm{var}(X) + \textrm{var}(Y) 
    \end{align}
    $$

## Unit 5: Continuous Random Variables

### Lecture 8: Probability density functions

* Probability density functions (PDFs) are the continuous counterpart of PMFs. 

* Notation: $f_X(x)$.

* A continuous random variable $X$ is any random variable where probabilities of events can be described by the formula

    $$
    \begin{align}
    \mathbf{P}(a \leq X \leq b) = \int\limits_a^b f_X(x)dx
    \end{align}
    $$

* A physical analogy of PDFs is as follows: For continuous random variables, the probability of events is now spread along the real line. To find the probability of an event, say $a \leq X \leq b$, we need to integrate over the interval to find the area under the curve. Now, since probabilities are dimensionless, a PDF of one variable must have dimensions of per unit length i.e. $f_X(x)$ is a probability per unit length as a function of $x$. 

* Note that for PDFs, the probability of any one point is defined by an integral with equal lower and upper limits, and hence this probability is zero. There it follows that

    $$
    \begin{align}
    \mathbf{P}(a \leq X \leq b) = \cancelto{0}{\mathbf{P}(X=a)} + \cancelto{0}{\mathbf{P}(X=b)} + \mathbf{P}(a<X<b) 
    \end{align}
    $$

    Therefore, probabilities of a closed interval is the same as probabilites of open intervals, and endpoints do not matter.

* PDFs follow similar rules as PMFs:

    $$
    \begin{align}
    f_X(x) & \geq 0 \\
    \int\limits_{-\infty}^{\infty} f_X(x) dx & = 1
    \end{align}
    $$

* All the results from the discrete cases port over to the continuous case:

    $$
    \begin{align}
    & \textrm{Expected value:} \mathbf{E}[X] = \int\limits_{-\infty}^{\infty}x f_X(x)dx \\
    & \textrm{Expected value rule:} \mathbf{E}[g(X)] = \int\limits_{-\infty}^{\infty}g(x) f_X(x)dx \\
    & \textrm{Linearity:} \mathbf{E}[aX + b] = a\mathbf{E}[X] + b \\
    & \textrm{var}(aX+b) = a^2 \textrm{var}(X)\\
    & \textrm{var}(X) = \mathbf{E}[X^2] - (\mathbf{E}[X])^2
    \end{align}
    $$

* **Exponential random variable:**

    $$
    \begin{align}
    f_X(x) = \begin{cases}
    \lambda e^{-x}, & \text{if } x \geq 0 \\
    0, & \text{otherwise}
    \end{cases}
    \end{align}
    $$

    Expected value: $1/\lambda$, variance: $1/\lambda^2$

* The exponential random variable is very similar to the discrete case of a geometric random variable. Think about the connection between the expected value and the variances of the two cases. Physically, the exponential random variable is used to model the time needed to wait for an event to occur: the probability that the time needed to wait for the occurrence of an event lies in an interval of large values is small. Think radioactive decay, etc. 

* The cumulative distribution function (CDF) helps us talk about continuous as well as discrete random variables using a single concept. The notation (capital letters) and definition is:

    $$
    \begin{align}
    F_X(x) = \mathbf{P}(X \leq x) =  \int\limits_{-\infty}^xf_X(x)dx
    \end{align}
    $$

    For discrete random variables, the integral becomes a sum.

* The CDF contains all the information I need about the PDF/PMF. From the fundamental theorem of calculus, it is easy to show that 

    $$
    \begin{align}
    \dfrac{dF_X(x)}{dx} = f_X(x)
    \end{align}
    $$

For the discrete case, the CDF will be a step function, and the height of the step at each $x_i$ gives us $p_X(x_i)$ i.e. 

    $$
    \begin{align}
    p_X(x_i) = F_X(x_i)-F_X(x_{i-1})
    \end{align}
    $$

* CDFs are monotonically increasing functions. Easy to prove.

* An important continuous PDF is the normal PDF, denoted by $N(\mu, \sigma^2)$ where $\mu$ and $\sigma^2$ are the mean and standard deviation of the distribution. The PDF is

    $$
    \begin{align}
    f_X(x) = \dfrac{1}{\sigma\sqrt{2\pi}}\exp\left({-\dfrac{(x-\mu)^2}{2\sigma^2}}\right)
    \end{align}
    $$

    The special case of $\mu=0$ and $\sigma^2=1$ is called the standard normal distribution, as opposed to the general normal distribution.

* The nice quality of the normal distribution is that if a random variable $X$ is normal, then $Y=aX+b$ is also normal. Therefore we can transform a general normal distribution of the form $X\sim N(\mu,\sigma^2)$ to a standard normal distribution $Y\sim N(0,1)$ by making the transformation $Y = \frac{X-\mu}{\sigma}$ and then use the standard normal tables to make probability calculations.

* $\mathbf{P}(X\leq \omega)$ is the same as $\Phi(\omega)$ where $\Phi(\omega)$ is the CDF. 

* The normal distribution has the property $\Phi(x) = 1-\Phi(-x)$.

### Lecture 9: Conditioning on an event; multiple random variables

* Conditioning in the continuous case (i..e when we talk about a PDF) is exactly the same concept as conditioning in the discrete case (i.e. when we talk about a PMF). The probabilities are renormalized such that the sum of all possible outcomes in the conditional world evaluates to 1. In other words, the relative probabilities inside the conditioned world remains the same, but everything is just scaled up.

* The definition of a conditional PDF is

    $$
    \begin{align}
    f_{X|X \in A}(x) &=\dfrac{f_X(x)}{\mathbf{P}(X \in A)}\\
    &=\dfrac{f_X(x)}{\int\limits_A f_X(x)dx}
    \end{align}
    $$

* Many of the other results derived previously for the discrete case remain the same:

    $$
    \begin{align}
    &\textrm{Expected Value: } \mathbf{E}[X|A] = \int_A x f_{X|A)(x)dx}\\
    &\textrm{Expected Value rule: } \mathbf{E}[g(X)|A] = \int_A g(x) f_{X|A}(x)dx
    \end{align}
    $$

* The exponential random variable is analogous to the geometric random variable. ONe can split up the domain into little section of length $\delta$ and the probability of achieving a ``success'' in the next interval/section is $\lambda\delta$. This turns out to be important for the Poisson process to be discussed later. 

* All of the discrete results port over:

    $$
    \begin{align}
    &\textrm{Total Probability Theorem: } f_X(x) = \sum_i\mathbf{P}(A_i) f_{X|A}(x) \\
    &\textrm{Total Expectation Theorem: }\mathbf{E}[X] = \sum_i\mathbf{P}(A_i)\mathbf{E}[X|A_i]
    \end{align}
    $$

    where all the $A_i$ are disjoint. 

* Mixed random variables are when you can neither classify the random variable as discrete or continuous. Consider the following random variable:

    $$
    \begin{align}
    X=\begin{cases}
    &\textrm{Uniform [0,2], with probability 1/2}\\
    &\textrm{1, with probability 1/2}
    \end{cases}
    \end{align}
    $$

    Now this is not discrete because there is a continuous random variable, and not continuous because the probability that $X$ takes the value 1 is finite (single point probabilities are 0 for a continuous random variable). Therefore this is mixed. It has neither a PDF or a PMF of its own.

* More generally, a mixed random variable can be thought of as being of the following form:

    $$
    \begin{align}
    X=\begin{cases}
    &Y, \textrm{ with probability } p\\
    &Z, \textrm{ with probability } 1-p
    \end{cases}
    \end{align}
    $$

    with $Y$ discrete and $Z$ continuous.

* These mixed random variables can be described nicely and concisely with a CDF. At each of the discrete cases, there is a `jump' in the CDF. Draw a few to see this.

* **Joint PDFs** are the to dimensional analog of PDFs in one dimension. The notation is $f_{X,Y}(x,y)$. Similar to the 1-D case, the joint PDF is defined as

    $$
    \begin{align}
    \mathbf{P}(a \leq X \leq b \textrm{ and } c \leq Y \leq d) = \int\limits_c^d \int\limits_a^b f_{X,Y}(x,y) dx dy 
    \end{align}
    $$

* Joint PDFs can be interpreted as probabilities per unit area. Therefore, any function of two variables that is also positive and respects that the sum of all probabilities in the domain evaluates to 1, is a valid joint PDF.

* Just like how the probability of any one point is 0 in the 1-D case, probabilities of events that lie on a line are zero in the joint case. 

* For a true joint PDF, it is not enough that both variables be continuous, but that the probabilities have to be truly spread over an area. Probability is not allowed to be concentrated over a 1-D set. 

* **Uniform joint PDF**: Find the constant by normalizing the PDF over the domain of non-zero probability. 

* Marginal probabilities are related to the joint distribution in a very similar fashion to the discrete case, with sums being replaced by integrals. 

* Expected value rule and linearity of expectations hold in this case too. 

* A joint CDF is defined in a very natural fashion too:

    $$
    \begin{align}
    F_{X,Y}(x,y) = \mathbf{P}(X \leq x, Y \leq y) = \int\limits_{-\infty}^x \int\limits_{-\infty}^y f_{X,Y}(u,v) du dv 
    \end{align}
    $$

    Moreover,

    $$
    \begin{align}
    f_{X,Y}(x,y) = \dfrac{\partial^2 F_{X,Y}}{\partial x \partial y}(x,y) 
    \end{align}
    $$

* Conditional PDFs are defined similarly to the discrete case, with some subtleties:

    $$
    \begin{align}
    f_{X|Y}(x|y) = \dfrac{f_{X,Y}(x,y)}{f_Y(y)}    
    \end{align}
    $$

    as long as $f_Y(y)>0$. We cal also define the probability of $X \in A$ given $Y=y$ as follows:

    $$
    \begin{align}
    \mathbf{P}(X \in A | Y=y) = \int_A f_{X|Y}(x|y) dy
    \end{align}
    $$

    The subtlety here is that the probability of an event $Y=y$ in the continuous case is 0, but we simply use the above equation as a definition and proceed without worrying too much about it. Proofs exist to show everything is nice and consistent and well-defined. 

* Total expectation theorem for a continuous PDF:

    $$
    \begin{align}
    \mathbf{E}[X] = \int\limits_y f_Y(y)\mathbf{E}[X|Y=y]dy
    \end{align}
    $$

* Independence of two continuous random variables are also described in the a similar fashion. Two random variables $X$ and $Y$ are independent if $f_{X,Y}(x,y) = f_X(x)f_Y(y)$. 

* Similar relationships for independence as in the discrete case hold:

    $$
    \begin{align}
    \mathbf{E}[XY] & = \mathbf{E}[X]\mathbf{E}[Y]\\
    \textrm{var}(X + Y) & = \textrm{var}(X) + \textrm{var}(Y)\\
    \mathbf{E}[g(X)h(Y)] & = \mathbf{E}[g(X)]\mathbf{E}[h(y)] 
    \end{align}
    $$

* Some intuition on normal distributions. For a joint standard normal PDF of two random variables $X$ and $Y$,

    $$
    \begin{align}
    f_{X,Y}(x,y) = \dfrac{1}{2\pi\sigma_x^2\sigma_y^2}\exp\left(-\dfrac{(x-\mu_x)^2}{2}-\dfrac{(y-\mu_y)^2}{2}\right)
    \end{align}
    $$

    The contours of this distribution (i.e. the locus of points of constant PDF are ellipses centered on $(\mu_x, \mu_y)$ with major and minor axes $\sigma_x$ and $\sigma_y$ respectively. 

* The Bayes rule for continuous random variables is very similar to the discrete case. It is instructive to derive it because it uses the concepts of multiplication rule and the total probability theorem. We first start with the definition of conditional probabilities as applied to the continuous variables:

$$
\begin{align}
f_{X,Y}(x,y) & = f_Y(y)\cdot f_{X|Y}(x|y)\\
& = f_X(x)\cdot f_{Y|X}(y|x)
\end{align}
$$

Equating the two and simplifying, we get

$$
\begin{align}
f_{X|Y}(x|y) = \dfrac{f_X(x)\cdot f_{Y|X}(y|x)}{f_Y(y)}
\end{align}
$$

and rewriting $f_Y(y)$ using the total probability theorem, we finally arrive at

$$
\begin{align}
f_{X|Y}(x|y) = \dfrac{f_X(x)\cdot f_{Y|X}(y|x)}{\int_x f_X(x')\cdot f_{Y|X}(y|x')dx'}
\end{align}
$$

* There also exists a **Mixed Bayes Rule** for cases where we have both a discrete random variable and a continuous random variable. This is quote common on real life situations, for example when we are interested in a discrete signal given that there is continuous noise. Let us assume that $K$ is a discrete random variable and $Y$ is a continuous random variable. Let us use the multiplication rule two ways:

    $$
    \begin{align}
    \mathbf{P}(K=k, y \leq Y \leq y+\delta) & = \mathbf{P}(K=k) \cdot \mathbf{P}(y \leq Y \leq y+\delta | K=k) \\
                        & = \mathbf{P}(y \leq Y \leq y+\delta) \cdot \mathbf{P}(K=k | y \leq Y \leq y+\delta)
    \end{align}
    $$

    We now rewrite the probabilities in PMF and PDF notation as appropriate.

    $$
    \begin{align}
    \mathbf{P}(K=k, y \leq Y \leq y+\delta) & = p_K(k)\cdot f_{Y|K}(y|k)\delta \\
                        & = f_{Y}(y)\delta \cdot p_{K|Y}(k|y) 
    \end{align}
    $$

    Equating these two and rearranging two ways, we get two different variations of the mixed Bayes rule:

    $$
    \begin{align}
    f_{Y|K}(y|k) & = \dfrac{f_Y(y) \cdot p_{K|Y}(k|y)}{p_K(k)} \\
    p_{K|Y}(k|y) & = \dfrac{p_K(k)\cdot f_{Y|K}(y|k)}{f_Y(y)}
    \end{align}
    $$

    Furthermore, we can use appropriate forms of the total probability theorem in each case to furher expand the to variations of Bayes rule above, i.e.,

    $$
    \begin{align}
    p_K(k) & = \int\limits_{y'} f_Y(y') \cdot p_{K|Y}(k|y')\ dy' \\
    f_Y(y) & = \sum_{k'} p_K(k') \cdot f_{Y|K}(y|k')
    \end{align}
    $$

* Can you recall the Monte-Carlo method? It's quite beautiful. 

## Unit 6: Further topics on random variables

### Lecture 11: Derived distributions

* Given a discrete random variable with PMF $p_X(x)$, a discrete random variable of the form $Y=aX+b$ has PMF $p_Y(y)=p_X\left(\frac{y-b}{a}\right)$.

* In the case of a continuous random variable $Y = aX +b$, more generally

    $$
    \begin{align}
    f_Y(y) = \dfrac{1}{|a|}f_X\left(\dfrac{y-b}{a}\right)
    \end{align}
    $$

    Intuitively, we shift the distribution by $b$, scale the values $X$ takes by $a$ and also scale the dsitribution itself by the magnitude of $a$ so that everything integrates to 1.

* For the very general case of $Y = g(X)$, we proceed using the two-step procedure (i) Find the CDF of $y$ and (ii) Differentiate to find the PDF. In other words (assuming X = h(Y)), 

    $$
    \begin{align}
    F_Y(y) & = \mathbf{P}(Y\leq y) \\
    & = \mathbf{P}(g(X) \leq y) \\
    & = \mathbf{P}(X \leq h(y)) \textrm{ (take care of direction of inequality)} \\
    \end{align}
    $$

    At this point, check the domains, etc. to evaluate $F_Y(y)$ and then differentiate to find $f_Y(y)$. 

* More generally, if $g(X)$ is a monotonically increasing or decreasing function, and if $x=h(y)$, then

    $$
    \begin{align}
    f_Y(y) = f_X(h(y))\cdot |h'(y)|
    \end{align}
    $$

* For variables consisting of multiple random variables (for example $Z=X/Y$), you are better off starting from first principles i.e. using the definition of the CDF.  

### Lecture 12: Sums of independent r.v's. Covariance and correlation

* **Convolution of two random variables:** Given two *independent* random variables $X$ and $Y$, and the sum of them $Z=X+Y$,

    $$
    \begin{align}
    f_Z(z) = \sum_xf_X(x)f_Y(z-x)
    \end{align}
    $$

    Think carefully about the allowed values of $x$ over which the sum is carried out. For the continuous case ($X$ and $Y$ are continuous random variables), the proof is quite instructive, so make sure you recall it. The general result is similar:

    $$
    \begin{align}
    f_Z(z) = \int\limits_x f_X(x)\cdot f_Y(z-x)dx
    \end{align}
    $$

    Note that the limits of allowable $x$'s are important. If for example, if $X$ and $Y$ can only take positive values, then $x:0\xrightarrow{}z$. Note the physical interpretation: convolution is an operation that yield the PDF of a sum of two independent random variables given their individual PDFs. 

* It is easy to show using the convolution formula that the sum of a finite number of normal random vairables is also normal. Specifically, if $X\sim N(\mu_x, \sigma_x^2)$ and $Y\sim N(\mu_y, \sigma_y^2)$, then $Z=X+Y \sim N(\mu_x+\mu_y,\sigma_x^2+\sigma_y^2)$.

* *Covariance of two random variables:* The covariance tells us in an average sense whether $X$ and $Y$ move together i.e. do they increas or decrease together. The definition is:

    $$
    \begin{align}
    \textrm{cov}(X,Y) = \mathbf{E}[(X-\mathbf{E}[X])\cdot (Y-\mathbf{E}[Y])]
    \end{align}
    $$

    If $X$ and $Y$ are independent, then the covariance is zero. However, if the covariance is zero, it does not mean that the random variables are independent.

* **Some properties of covariances:**

    $$
    \begin{align}
    \textrm{cov}(X,X) & = \textrm{var}(X)\\
    \textrm{cov}(X,Y) & = \mathbf{E}[XY]-\mathbf{E}[X]\mathbf{E}[Y]\\
    \textrm{cov}(aX+b,Y) & =a \cdot \textrm{cov}(X,Y)\\
    \textrm{cov}(X,Y+Z) & = \textrm{cov}(X,Y)+\textrm{cov}(X,Z) 
    \end{align}
    $$

* **Further general properties of covariances:**

    $$
    \begin{align}
    \textrm{cov}(X_1+X_2) & = \textrm{var}(X_1) + \textrm{var}(X_2) + 2\textrm{cov}(X1,X2) \\
    \textrm{var}(X_1+X_2+\cdots+X_n) & = \sum_i \textrm{var}(X_i) + \sum_{(i,j):i \neq j}\!\!\!\!\! \textrm{cov}(X_i,X_j)
    \end{align}
    $$

    Note that if $X_i$ are independent, then the variance of a sum of random variables is the sum of the variances.

* **The correlation coefficient:** This nondimensionalizes the covariances and hence is more helpful in interpreting the magnitude of correlation between two random variables. This is defined as

    $$
    \begin{align}
    \rho(X,Y) = \dfrac{\textrm{cov}(X,Y)}{\sigma_X \sigma_Y}
    \end{align}
    $$

* It can be shown that $-1\leq\rho\leq 1$. For perfect correlation, $\rho=1$. This means that the two random variables are related through some deterministic equation. For independent random variables, $\rho=0$. 

* Property: $\rho(aX+b,Y) = \textrm{sign}(a)\cdot\rho(X,Y)$

### Conditional expectation and variance revisited; Sum of a random number of independent random variables

* A conditional expectation $\mathbf{E}[X\vert Y]$ can be viewed as a function because it's value depends on the specific value of $Y$. Therefore, we can define $g(y) = \mathbf{E}[X\lvert Y=y]$. Now when we inspect the quantity $g(Y)$, it itself is a random variable that maps $Y=y$ to some particular number. More formally, we can view conditional expectation as random variables by defining

    $$
    \begin{align}
    g(Y) = \mathbf{E}[X|Y]
    \end{align}
    $$

    This has all the properties of random variables i.e. it has a mean, variance, expectation, etc. 

* **The law of iterated expectations:** $\mathbf{E}[\mathbf{E}[X\lvert Y]] = \mathbf{E}[X]$. This is quite easy to show using either first principles, or the total expectation theorem. Note that $\mathbf{E}[X\lvert Y=y]$ is a different quantity than $\mathbf{E}[X\lvert Y]$. The former is a number, because the number $Y=y$ is given, while the latter is a function, i.e., a random variable. 

* We can use similar ideas to think of the conditional variance as a random variable. If we define the quantity $\textrm{var}(X\lvert Y)$, this quantity can be interpreted as a random variable that takes the value $\textrm{var}(X\lvert Y=y)$ when the random variable $Y=y$. 

* The iterated expectation theorem does not hold here. In fact, the \textbf{the law of total variance} is

    $$
    \begin{align}
    \textrm{var}(X) = \mathbf{E}[\textrm{var}(X|Y)]+\textrm{var}(\mathbf{E}[X|Y])
    \end{align}
    $$


* Both of the law of iterated expectations and the total variance theorem are very useful in divide and conquer approaches. Some intuition here: when we are given data is \textit{groups} i.e. we split all the data into some $i$ groups, and then we want statistics on the pooled data for the whole range of $i$, these theorems are very useful. 

* One application of the iterated expectation theorem is the expected value of a sum of a random number of independent random variables. If $N$ is a random variables denoted the number of random variables $X_i$, then if we define $Y = X_1 + \cdots + X_n$, $\mathbf{E}[Y] = \mathbf{E}[N]\cdot\mathbf{E}[X_i]$ if all the $X_i$'s are drawn from the same distribution (i.e. have the same mean).

## Unit 7: Bayesian Inference

### Lecture 14: Introduction to Bayesian Inference

* At its heart, inference is the process of evaluating the value of a certain random variable given the observation of the values of other related random variables. 

* In Hypothesis testing problems, we have a small number of discrete models and we are trying to determine which model best describes some real world event. In estimation problems, we are trying to determine the value of a numerical quantity as close as possible to the real (unknown) value. 

* Conceptually, this is what is happened. Assume that we want to determine the model $\Theta$. For simplicity assume you are trying to fir a straight line $\Theta = aX + b$. Because we don't know the value of $a$ and $b$, $\Theta$ is at this point a random variable. Now, we have some \textit{a priori} assumption for the PDF of $\Theta$ which we call to be $f_\Theta(\theta)$. Now let us say that we have another random variable $X$ that takes on certain values that will help us determine $f_\Theta(\theta)$. Therefore, we can define a conditional probability distribution $f_{\Theta \lvert  X}(\cdot \lvert  X)$. Now, we carry out an experiment that results in $X=x$. We now do a posterior calculation using the Bayes rule to determine $f_{\Theta \lvert X}(\cdot \lvert  x)$. 

* Sometimes we may want to characterize the goodness of the determination of the posterior probability using a single number. For this, we can turn to the maximum \textit{a posteriori} probability (MAP) estimator, which is given by

    $$
    \begin{align}
    f_{\Theta\lvert X}(\theta^*\lvert x) = \max\limits_\theta f_{\Theta \lvert  X}(\theta \lvert  x)
    \end{align}
    $$

    or using the conditional expectation estimator, otherwise called the least-mean-square error. More on this later. 

* $\hat{\Theta} = g(X)$ is called the estimator, while $\hat{\theta} = g(x)$ is called the estimate. Think in terms of fitting a straight line.

* Some more insight on the MAP rule: Let us assume that we started with the prior $p_\Theta(\theta)$, made the observation $X=x$, and determined the a posteriori PMF $p_{\Theta\lvert X}(\theta\lvert x)$. Using the MAP rule, we pick that value of $\theta(=\hat{\theta})$ where $p_{\Theta\lvert X}(\theta\lvert x)$ is maximum. Now, to calculate how good this choice is (\textit{the probability of error}), we calculate $\mathbf{P}(\hat{\theta} \neq \Theta \lvert X=x)$. We can infact now extend this to all $x$'s or $\theta$'s to find (using the total probability theorem):

    $$
    \begin{align}
    \mathbf{P}(\hat{\Theta} \neq \Theta) & = \sum_{x'} \mathbf{P}(\hat{\Theta} \neq \Theta \lvert  X=x)\cdot p_X(x) \\
    & = \sum_{\Theta'} \mathbf{P}(\hat{\Theta} \neq \Theta \lvert  \Theta=\theta')\cdot p_\Theta(\theta')
    \end{align}
    $$

    Using the MAP rule, the overall probability of error is the optimum way to do hypothesis testing, because each term in the sum is minimized, so the entire sum is minimized. Choose whichever one is easier to calculate the probability of error. Often its the second formula because there are usually only a handful of discrete cases to calculate. 

* $\hat{\Theta}_{LMS} = \mathbf{E}[\Theta\lvert X]$

* In general, we can calculate the expected value of the error to see how well our estimator does by calculating $\mathbf{E}[\hat{\Theta}-\Theta\lvert X=x]$ which would be based on the posterior distribution, or we could calculate it independent of observation by directly calculating $\mathbf{E}[\hat{\Theta}-\Theta]$. 

* We can summarize the problem of Bayesian Inference as follows: we have some random variable $\Theta$ (say the bias of a coin) that we wish to determine. We first assume a prior distribution $f_\Theta(\theta)$. We then carry out an experiment described by the random variable $X$ whose outcome depends on what $f_\Theta(\theta)$ is (say we flip the coin 10 times). So now we have the quantity $p_{X\lvert \Theta}(x\lvert \Theta)$. We can now use Bayes rule to update our prior i.e. to get the posterior distribution:

    $$
    \begin{align}
    f_{\Theta \lvert  X}(\theta\lvert x) = \dfrac{p_{X\lvert \Theta}(x\lvert \theta)\cdot f_\Theta(\theta)}{p_X(x)}
    \end{align}
    $$

    We can run multiple experiments to update our prior sequentially to arrive at closer and closer estimates for $f_\Theta(\theta)$. To go from the distribution to the actual bias of the coin $\hat{\theta}$, we use MAP or the conditional expectation for the distribution, or perhaps some other suitable metric. 

### Lecture 15: Linear models with normal noise

Here we explore in detail models of the form

$$
\begin{align}
X_i = \sum\limits_{j=1}^m a_{ij}\Theta_j + W_i
\end{align}
$$

where some of the $\Theta_i$ are unknown, need to be determined a posteriori, and the $W_i$ are normal noise. These sorts of Bayesian inference problems occur extremely commonly in various physical situations. 

* It is easy to show that a function of the form

    $$
    \begin{align}
    c\cdot \exp[-(\alpha x^2 + \beta x + \gamma)]
    \end{align}
    $$

    also represents the PDF of a normal random variable as long as $\alpha>0$. Just complete squares. 

* Consider the linear case, where $X = \Theta + W$. Let us assume that $W\sim N(\mu_W,\sigma_W^2)$ and that the prior distribution for $\Theta$ is $\Theta\sim N(\mu_\Theta,\sigma_\Theta^2)$, and with $W$ and $\Theta$ being independent. Now we wish to find the posterior probability $f_{\Theta\lvert X}(\theta\lvert x)$. This would involve the application of the Bayes rule to find the posterior probability, but we first need to find $f_{X\lvert \Theta}(x\lvert \theta)$. We can do this as before by starting with the CDF, making appropriate substitutions, we can show that

    $$
    \begin{align}
    f_{X\lvert \Theta}(x\lvert \theta) = f_W(x-\theta)
    \end{align}
    $$

    Now, we apply the Byes rule to find

    $$
    \begin{align}
    f_{\Theta\lvert X}(\theta\lvert x) & = \dfrac{f_\Theta(\theta) \cdot f_{X\lvert \Theta}(x\lvert \theta)}{f_X(x)} \\
    & = \dfrac{1}{f_X(x)}\cdot c_1 \exp\left[-\frac{1}{2\sigma_\Theta^2}(\theta-\mu_\theta)^2\right]\cdot
    c_2 \exp\left[-\frac{1}{2\sigma_W^2}(x-\theta-\mu_W)^2\right]
    \end{align}
    $$

    To find the MAP estimate, we minimize the posterior probability w.r.t. $\theta$ and after some algebra we find that

    $$
    \begin{align}
    \hat{\theta} = \dfrac{\sigma_W^2 \mu_\theta+\sigma_\theta^2(x-\mu_W)}{\sigma_\theta^2+\sigma_W^2}
    \end{align}
    $$

* Consider a more general case where we have some unknown random variable $\Theta$ that we are trying to estimate. But to do this estiamtion, we can make a series of measurements $X_i$ that are corrupted by normally distributed noise. That is, $X_i = \Theta + W_i$, $i=1, \ldots, n$, where $W_i$'s are standard normals with variance $\sigma_i^2$. Our prior for $\Theta$ is assumed to be $\Theta \sim N(\mu_0, \sigma_0^2)$. The task is to estimate the value of $\Theta$ based on these multiple observations of $X_i$. We again turn to using Bayesian inference, but now, whenever we refer to $X$, we real mean the vector $X=(X_1, X_2, \ldots, X_n)$. The Bayes theorem gives us as before

    $$
    \begin{align}
    f_{\Theta\lvert X} = \dfrac{f_{\Theta}(\theta)\cdot f_{X\lvert \Theta}(x\lvert \theta)}{f_x(x)} \label{eqn:MultipleInference}
    \end{align}
    $$

    We now need to find $f_{X\lvert \Theta}(x\lvert \theta)$. The key observation here is that all the $W_i$ are independent, and hence, given the value of $\theta$, all the $X_i$'s are also independent normal modes whose mean is now shifted to value $\theta$. Therefore,

    $$
    \begin{align}
    f_{X\lvert \Theta}(x\lvert \theta) & = f_{X_1, X_2, \ldots, X_n\lvert \Theta}(x_1,x_2, \dots, x_n\lvert \theta) \\ 
    & = \prod\limits_{i=1}^n f_W(x_i-\theta)
    \end{align}
    $$

    We now have everything we need to plug into equation~\eqref{eqn:MultipleInference}. After some algebra, and minimization, we find that:

    $$
    \begin{align}
    \hat{\theta} = \dfrac{\sum\limits_{i=0}^n\dfrac{x_i}{\sigma_i^2}}{\sum\limits_{i=0}^n\dfrac{1}{\sigma_i^2}}
    \end{align}
    $$

* Note: more generally, $X_i=c_i \Theta+W_i$. In this case, the mean of $f_{X\lvert \Theta}(x_i\lvert \theta)$ gets shifted to $c_i\theta$ but everything else remains the same. In general, it's best to follow first principles of starting with the CDF when in doubt trying to relate the random variables. 

* One metric to see how well the estimator is doing is based on calculating the mean squared error $\mathbf{E}[(\Theta-\hat{\Theta})^2\lvert X=x]$. Given that $X=x$, $\Hat{\Theta}$ is completely determined to be $\Hat{\theta}$. Therefore, $\mathbf{E}[(\Theta-\hat{\Theta})^2\lvert X=x] = \mathbf{E}[(\Theta-\Hat{\theta})^2\lvert X=x]$. But it can be shown easily that $\Hat{\theta} = \mathbf{E}[\Theta\lvert X]$ (we have discussed this previously), so $\mathbf{E}[(\Theta-\Hat{\theta})^2]$ is just the variance of $\Theta\lvert X$. This is a key point, the mean-square-error is the variance of the conditional posterior distribution. To find the total mean-square-error, we can use the law of iterated expectation to see that $\mathbf{E}[(\Theta-\Hat{\Theta)}^2]=\mathbf{E}[\mathbf{E}[\Theta-\Hat{\Theta}\lvert X]] = \mathbf{E}[\textrm{var}(\Theta\lvert X)]$. If both $\Theta$ and $W_i$ are normal, then we can easily calculate this through completion of squares. It turns out that 

    $$
    \begin{align}
    \textrm{var}(\Theta\lvert X) = \left(\sum\limits_{i=0}^n\dfrac{1}{\sigma_i^2}\right)^{-1}
    \end{align}
    $$

* In the case of multiple variables, say for example trajectory estimation problems, where we have $X_i = \Theta_0+\Theta_1t_i+\Theta_2t_i^2$, everything remains the same, except that whenever we write $\Theta$, we mean the vector $\Theta = (\Theta_0,\Theta_1,\Theta_2)$ an whenever we write $X$, we mean the vector $X=(x_1, X_2, \ldots, X_n)$. Now when we write our Bayes, theorem, the prior probability is given by

    $$
    \begin{align}
    f_\Theta(\theta) = \prod\limits_{i=0}^2 f_{\Theta_j}(\theta_j)
    \end{align}
    $$

    which can be proved because all the $\Theta_i$ are independent. Similarly, $f_{X\lvert \Theta}(x,\theta)$ can be shown by the independence property to be

    $$
    \begin{align}
    f_{X\lvert \Theta}(x\lvert \theta) = \prod\limits_{i=1}^n f_{X_i\lvert \Theta}(x_i\lvert \theta)
    \end{align}
    $$

    where

    $$
    \begin{align}
    f_{X_i\lvert \Theta}(x_i\lvert \theta) \sim N(\theta_0+\theta_1 t_i + \theta_2 t_i^2,\sigma^2)
    \end{align}
    $$

    We then plug in everything into the Bayes theorem, and to find the values of $\hat{\theta}_1, \hat{\theta}_2, \hat{\theta}_3$, we minimize with respect to each $\Theta_i$ and so we have 3 equations and 3 unknowns. 

### Lecture 16: Least Mean Squares (LMS) estimation

* When we talk about minimizing the mean square error, what we mean is that we want to minimize $\mathbf{E}[(\Theta-\hat{\theta})^2]$. If we expand that square expression and then minimize w.r.t. $\hat{\theta}$, we arrive at $\hat{\theta}=\mathbf{E}[\Theta]$. This is why when we say that when our estimate is $\mathbf{E}[\Theta]$ we call it the mean square estimate. 

* Given this fact, we can see that $\textrm{var}(\Theta) = \mathbf{E}[(\Theta-\mathbf{E}[\Theta])^2]$ which is the same as the mean square error. 

* $\hat{\theta} = \mathbf{E}[\Theta]$ minimizes the mean square error $\mathbf{E}[(\Theta-\hat{\theta})^2]$. In other words,

    $$
    \begin{align}
    \mathbf{E}[(\Theta-\mathbf{E}[\Theta])^2] \leq \mathbf{E}[(\Theta-c)^2]
    \end{align}
    $$

    for any $c$. In a conditional universe, we can write this as

    $$
    \begin{align}
    \mathbf{E}[(\Theta-\mathbf{E}[\Theta\lvert X=x])^2\lvert X=x] \leq \mathbf{E}[(\Theta-g(x))^2\lvert X=x]
    \end{align}
    $$

    for any $x$. More abstractly, we can write this in terms of random variables as

    $$
    \begin{align}
    \mathbf{E}[(\Theta-\mathbf{E}[\Theta\lvert X])^2\lvert X] \leq \mathbf{E}[(\Theta-g(X))^2\lvert X]
    \end{align}
    $$

    Now if we take the expected value on both sides, and use the law of iterated expectations, we have

    $$
    \begin{align}
    \mathbf{E}[(\Theta-\mathbf{E}[\Theta\lvert X])^2] \leq \mathbf{E}[(\Theta-g(X))^2]
    \end{align}
    $$

    This tells us that the estimator $\mathbf{E}[\Theta\lvert X]$ minimizes the least square error compared to any other estimator $g(X)$. 

* What is the value of the mean square error? 

    $$
    \begin{align}
    \mathbf{E}[(\Theta-\mathbf{E}[\Theta])^2] = \textrm{var}(\Theta)
    \end{align}
    $$

    so the value of the mean square error is just the variance of $\Theta$. If we w ere in a conditional universe,

    $$
    \begin{align}
    \mathbf{E}[(\Theta-\mathbf{E}[\Theta\lvert X=x])^2\lvert X=x] = \textrm{var}(\Theta\lvert X=x)
    \end{align}
    $$

    More abstractly,

    $$
    \begin{align}
    \mathbf{E}[(\Theta-\mathbf{E}[\Theta\lvert X])^2\lvert X] & = \textrm{var}(\Theta\lvert X) \\
    \Rightarrow \mathbf{E}[(\Theta-\mathbf{E}[\Theta\lvert X])^2] & = \mathbf{E}[\textrm{var}(\Theta\lvert X)]
    \end{align}
    $$

    where we have used the law of iterated expectations in the second step. Therefore, the average mean square error is the expected value of the conditional variance. To actually carry out that calculation, it can be done through seeing that the expected value is an average and therefore

    $$
    \begin{align}
    \mathbf{E}[\textrm{var}(\Theta\lvert X)] = \int\limits_x f_X(x)\textrm{var}(\Theta\lvert X=x)dx
    \end{align}
    $$

* In summary, the conditional LMS estimate is the value that minimizes $\mathbf{E}[(\Theta-\hat{\Theta})^2\lvert X]$ which turns ot to be $\mathbf{E}[\Theta\lvert X]$. The error in the conditional LMS is the conditional variance $\textrm{var}(\Theta\lvert X)$. To find the unconditional error, we wish to find $\mathbf{E}[(\hat{\Theta}-\Theta)^2]$ which can be found from an application of the law of iterated expectations to obtain $\mathbf{E}[\textrm{var}(\Theta\lvert X)]$. 

* The MAP estimate works well for hypothesis testing problems. An LMS estimator does not work for hyothesis testing problems.

* If $X=(X_1, X_2, \ldots, X_n)$ were a vector, everything remains the same and the conditional LMS estimate is given by $\mathbf{E}[\Theta\lvert X_1=x_1, X_2=x_2, \ldots, X_n=x_n]$. 

* If $\Theta = (\Theta_1, \Theta_2, \ldots, \Theta_n)$ were a vector, then things are also very similar. In this case, we simply find each optimum $\hat{\Theta}_j = \mathbf{E}[\Theta_j\lvert X_1=x_1, X_2=x_2, \ldots, X_n=x_n]$. However, calculating this is not always easy. To find the RHS,

    $$
    \begin{align}
    \mathbf{E}[\Theta_j\lvert X=x] = \int\limits_{\theta_j} \theta_j f_{\Theta_j\lvert X}(\theta_j\lvert x) d\theta_j
    \end{align}
    $$

    and $f_{\Theta_j\lvert X}(\theta_j\lvert x)$ is itself the marginal PDF, i.e., an integral of the joint PDF $f_{\Theta\lvert X}(\Theta\lvert x)$ over all $\theta_i \neq \theta_j$ 

* If we define the error $\Tilde{\Theta} = \hat{\Theta}-\Theta$, then $\mathbf{E}[\Tilde{\Theta}] = \mathbf{E}[\hat{\Theta}-\Theta] = \mathbf{E}[\mathbf{E}[\Theta]-\Theta] = 0$. In fact, something more is true. It can be shown that $\mathbf{E}[\hat{\Theta}-\Theta\lvert X=x]=0$. Can you prove this?

* Another property: $\textrm{cov}(\Tilde{\Theta},\hat{\Theta}) =0$ and therefore, $\textrm{var}(\Tilde{\Theta}+\hat{\Theta}) =  \textrm{var}(\Tilde{\Theta})+\textrm{var}(\hat{\Theta}) $

### Lecture 17: Linear Least Mean Squares (LLMS) Estimation

* If our objective is to keep the mean squared error $\mathbf{E}[(\Theta-\hat{\Theta})^2\lvert X]$ small, then the best possible estimator is the conditional expectation $\hat{\Theta} = \mathbf{E}[\Theta\lvert X]$. The total mean squared error $\mathbf{E}[(\Theta-\hat{\Theta})^2]$ itself is $\mathbf{E}[\textrm{var}(\Theta\lvert X)]$.  

* Sometimes, it can be hard to calculate the conditional expectation, or maybe we are missing some details of the various probability distributions. In these cases, we only seek estimators that are of the form $\hat{\Theta} = aX+b$. 

* In generation, the LMS estimator tells us that $\hat{\Theta}=\mathbf{E}[\Theta\lvert X]$ i.e., find the $\hat{\Theta}$ that minimizes $\mathbf{E}[(\Theta-\hat{\Theta})^2\lvert X]$. In linear least means square estimation, we impose the constraint $\hat{\Theta} = aX+b$, and hence we are looking for the values of $a$ and $b$ that minimize $\mathbf{E}[(\Theta-aX-b)^2]$. 

* **Development of the solution to the linear least mean squares estimate**:

    We know that we want to miinimize $\mathbf{E}[(\Theta-aX-b)^2]$ w.r.t $a,b$. For this, we will assume that that we have already fixed $a$ and focus on finding $b$ first. We essentially have a set of linear equations from $\partial \mathbf{E}[(\Theta-aX-b)^2]/\partial a$ and $\partial \mathbf{E}[(\Theta-aX-b)^2]/ \partial b$. 

    $$
    \begin{align}
    \dfrac{\partial}{\partial b}\mathbf{E}[((\Theta-aX)-b)^2] & = \dfrac{\partial}{\partial b} (\E[(\Theta-aX)^2+b^2-2b(\Theta-aX)])\\
    & = \dfrac{\partial}{\partial b} (\E[(\Theta-aX)^2]+b^2-2b\E[\Theta-aX])\\
    \end{align}
    $$

    Doing all the algebra, and realizing some of he random variables are constants w.r.t. $b$, we get

    $$
    \begin{align}
    b = \E[\Theta-aX]
    \end{align}
    $$

    Now, we can plug this value of $b$ into $\mathbf{E}[(\Theta-aX-b)^2]$ we find

    $$
    \begin{align}
    \mathbf{E}[(\Theta-aX-b)^2] & = \mathbf{E}[(\Theta-aX)-(\E[\Theta-aX]))^2] \\
    & = \textrm{var}(\Theta-aX) \\
    & = \textrm{var}(\Theta)+a^2\textrm{var}(X)-2a\textrm{cov}(\Theta,X) \\
    \Rightarrow a & = \dfrac{\textrm{cov}(\Theta,X)}{\textrm{var}(X)}
    \end{align}
    $$

    We can further simplify the covariance in terms of the correlation co-efficient:

    $$
    \begin{align}
    \rho = \dfrac{\textrm{cov}(\Theta,X)}{\sigma_\Theta \sigma_X}
    \end{align}
    $$

    So after all the algebra, we find two forms:

    $$
    \begin{align}
    \hat{\Theta}_L  & = \E[\Theta]+\dfrac{\textrm{cov}(\Theta,X)}{\textrm{var}(X)}(X-\E[X]) \\
    \hat{\Theta}_L  & = \E[\Theta]+\rho\dfrac{\sigma_\Theta}{\sigma_X}(X-\E[X]) \\
    \end{align}
    $$

    Note the subscript $L$ a denoting linear least squares estimator.

* It is easy to show that the mean square error $\E[(\Theta-\hat{\Theta}_L)^2] = (1-\rho^2)\textrm{var}(\Theta)$. Intuitively, this means that when we start out, we just have the variance of $\Theta$, but if $\rho \neq 0$, we get more certainty about the variance of $\Theta$ and hence the linear least mean square error drops. 

* To actually calculate the forms of the linear estimator $\hat{\Theta}_L$, there is a lot of plus ang chug involving calculating the expectations, variances and covariances of $\Theta$ and $X$. 

* This approach to find the LLMS estimator also works when there are multiple observations. In that case, we define $\hat{\Theta} = \sum_i a_i X_i + b$ and we wish to minimize $\E[(\sum_i a_iX_i -\Theta)^2]$ w.r.t. the $a_i$'s and $b$. The process is the same - we are concerned with the various expectations, variances and covariances of the random variables $\Theta$ and $X_i$, although the final expression is uglier.

* Couple of important points: (1) If $\hat{\Theta}\_{LMS}$ is linear w.r.t. $X$ say this is obtained from the conditional expectation), then this is the best possible linear estimate as well, and hence $\hat{\Theta}\_{LMS}=\hat{\Theta}\_{LLMS}$. (2) $\hat{\Theta}_{LLMS}$ only cares about means, expectations and variances of the random variables involved, and hence does not really care about the details of the prior distributions of the random variables. 

* When we talk about Linear Least Mean Squared estimators, we are really concerned withthe fact that the coefficients in the relationship between $\Theta$ and the other random variables are linear. So, it is also legal to define some estimator as being, for example, 

    $$
    \begin{align}
    \hat{\Theta} = a_1 X + a_2 X^3 + a_3 \exp(X) + a_4 \ln(X) + b
    \end{align}
    $$

    Any functions $g(X)$ would work because we can just redefine $X_1=X$, $X_2=X^3$, $X_3 = \exp(X)$, etc. However, the trade-off here is that the expectations, variances and covariances of these variables become increasingly complicated, so these choices now become less about the mathematical details of the problem, which is procedural, but more about domain-specific knowledge. 

## Unit 8: Limit Theorems and Classical Statistics

### Inequalities, convergence and the weak law of large numbers

* **Markov Inequality:** If $X \geq 0$, then

    $$
    \begin{align}
    \mathbf{P}(x \geq a) \leq \dfrac{\E[X]}{a}
    \end{align}
    $$

    There is a really clever proof which starts by defining a new random variable $Y = 0$ if $X <a$ and $Y = a$ if $X \geq a$. Can you follow this line of thought?

    Physically, the Markov Inequality tells us that if the expected value of $X$ is small, then the probability that $X$ takes large values is small.

* **Chebychev Inequality:** If $X$ has mean $\mu$ and variance $\sigma^2$, then

    $$
    \begin{align}
    \mathbf{P}(\lvert X-\mu\lvert >c) \leq \dfrac{\sigma^2}{c^2}
    \end{align}
    $$

    i.e. it bounds the probability of being too far away from the mean. Physically, this inequality tells us that if the variance is small, then we don't expect the random variable to be too far away from the mean. The Chebychev inequality often performs better than the Markov inequality because it captures some information about the variance as well. It can be proved pretty easily using the Markov inequality. 

* **The weak law of large numbers:** Consider $X_1, X_2, \ldots, X_n$ to be $n$ independent, identically distributed (i.i.d) random variables all sampled/drawn from the same distribution of mean $\mu$ and variance $\sigma^2$. Define a new random variable called the sample mean

    $$
    \begin{align}
    M_n=\dfrac{X_1 + X_2 + \cdots + X_n}{n}
    \end{align}
    $$

    then, the weak law of large numbers states that

    $$
    \begin{align}
    \mathbf{P}(\lvert M_n -\mu\lvert  > \varepsilon) \leq \dfrac{\sigma^2}{n\varepsilon^2}
    \end{align}
    $$

    The proof is a fairly simple application of Chebychev's inequality, and by noting that all the $X_i$'s are independent and hence variance of the sum is the sum of the variances.

    From this, we can see that as $n \to \infty$ and fixed $\varepsilon$, the probability that the sample mean deviates from the true mean becomes arbitrarily small. 

* **Definition of convergence in probability:** The notion of convergence in probability is somewhat similar to convergence of real numbers; it also closely follows the weak law of large numbers in the large $n$ limit. The notion of convergence in probability states that

    $$
    \begin{align}
    \mathbf{P}(\lvert Y_n - a\lvert >\varepsilon) \to 0 \textrm{ as } n \to \infty
    \end{align}
    $$

    Similar to the definition of real numbers, this definition says that for some large enough $n$ all the probability mass of $Y_n$ falls within $a-\varepsilon$ and $a+\varepsilon$ for any $\varepsilon > 0$. Convergence in probability basically says that \textit{the probability} that the random variable deviates too far out from $a$ tends to zero. But this might happen very slowly so the expected value itself might be very large (expected value is the area under the curve, which might not necessarily converge). 

    In other words, the probability that the random variable can deviate from the number $a$ is small, but the value of the random variable itself can be large, and hence the point below. 

    Note that if $Y_n$ converges to $a$, it does not necessarily mean that $\E[Y_n] \to a$.

* There are other notions of convergence, for example the strong law of large numbers which is defined by the notion of *convergence with probability 1*. This is a subtle and advanced topic, and not considered in this class.  

### Lecture 19: The central limit theorem (CLT)

* Consider the random variables $X_i$ to be i.i.d, with mean $\mu$ and variance $\sigma^2$ and consider the random variable $S_n = X_1 + \cdots + X_n$. Now construct the random variable

    $$
    \begin{align}
    Z_n=\dfrac{S_n - n\mu}{\sigma \sqrt{n}}
    \end{align}
    $$

    The central limit theorem states that

    $$
    \begin{align}
    \lim_{n \to \infty}\mathbf{P}(Z_n \leq z) = \mathbf{P}(Z \leq z)
    \end{align}
    $$

    where $Z$ is a standard normal random variable. In other words, the CDF of $Z_n$ approaches $\Phi(z)$ as $n \to \infty$. This is very deep, intuitive and nontrivial result. It doesn't matter what the exact distribution of the $X_i$'s are. 

* How large an $n$ do we need? This depends on how close the distribution of $S_n$ is to a normal random variable. FOr example, if the $X_i$'s are normal then there is no approximaion. Te result is exact. In all other cases, it helps if the distribution is unimodal (single peak), and other such features. 

* The basic types of problem we can solve using the CLT is of the form $\mathbf{P}(S_n \geq a) \leq b$ and you are given two of the three quantities $n, a, b$ and you're trying to find the third. Sometimes you will have to convert the problem statement to this form and this conversion may not always be obvious. It involves some manipulation of the variables. 

* Here is an interesting example: consider $X_1, \ldots, X_m$ to be $n$ Bernoulli random variables with parameter $p$. Then, $S_m = X_1 + \cdots X_m$ is Binomial with parameters $n, p$ (Go back to thinking about this in terms of indicator random variables). Now, we ask the question what is $\mathbf{P}(S_m \leq 21)$. 

    The first thing we do is to calculate the mean and variance of the binomial which are $np$ and $np(1-p)$ respectively. Therefore, 

    $$
    \begin{align}
    \mathbf{P}(S_m \leq 21) = \mathbf{P}\left(\dfrac{S_m - np}{np(1-p)\sqrt{m}} \leq 21\right) \approx \Phi(21)
    \end{align}
    $$

    from an application of the central limit theorem. Interestingly, this also offers an approach to calculate binomial probabilities. For example, if I want to calculate $\mathbf{P}(S_m = 19)$, I could of course do $\mathbf{P}(18 \leq S_m \leq 20)$. and appropriately normalize and translate to use the CLT. But because the CLT is continuous, I get a better approximation if I do $\mathbf{P}(18.5 \leq S_m \leq 19.5)$ and proceed as usual using the CLT approximation. 

### Lecture 20: An introduction to classical statistics

* In Bayesian inference, we have some unknown parameter $\theta$ that we model as a random variable $\Theta$ and hence we have a prior distribution $f_\Theta(\theta)$. We then looks at some other random variable $X$ that depends on $\Theta$ and we then use the Bayes rule to find the posterior conditional distribution $f_{\Theta\lvert X}(\theta\lvert x)$. The big difference in classical statistics is that we do not model the parameters as being an unknown random variable. It is modeled as simply being an \textit{constant} $\theta$ that is unknown. Therefore, we do not have access to the Bayes rule. Instead all we have is the distribution $f_X(x; \theta)$. Note the difference in notation: here there is no conditional distribution. Instead, we have some data, we generate multiple distributions $f_X(x; \theta)$ corresponding to different specific values of $\theta$ and see which value of the parameter best explains the observed data. 

* When we come up with estimators $\hat{\Theta}_n$, we chiefly want the estimator to have two properties: (1) It is unbiased, i.e., $\E[\hat{\Theta}_n] = \theta$ and (2) that it is consistent, which means that the weak law of large numbers holds and that as $n \to \infty$, $\E[\hat{\Theta}_n]$ approaches $\theta$. In other words, $\hat{\Theta}_n$ converges in probability to $\theta$.

* An estimator $A$ for a parameter that we are trying to estimate $a$ is unbiased if $\E[A] = a$. 

* We measure the performance of an estimator by looking at the mean squared error $\E[(\hat{\Theta} - \theta)^2]$. We can rewrite this as 

    $$
    \begin{align}
    \E[(\hat{\Theta} - \theta)^2] & = \textrm{var}(\hat{\Theta}-\theta)+(\E[\hat{\Theta}-\theta])^2 \\
                & = \textrm{var}(\hat{\Theta})+(\textrm{bias})^2
    \end{align}
    $$

    Try examining what each term is for two kinds of estimators, and what it means: (1) the sample mean $\hat{\Theta}_n$, and another somewhat silly estimator $\hat{\Theta} = 0$. The mean-squared-error for $\hat{\Theta}_n$ is the same as $\textrm{var}(\hat{\Theta}_n)
    = \sigma^2/n$. It is more common to report $\sqrt{\textrm{var}(\hat{\Theta}_n)
    } = \sigma/\sqrt{n}$ which is called the standard error. 

* **Confidence Interval:** A confidence interval of $1-\alpha$ is the interval $[\hat{\Theta}^-, \hat{\Theta}^+]$ such that

$$
\begin{align}
\mathbf{P}(\hat{\Theta}^- \leq \theta \leq \hat{\Theta}^+) \geq 1-\alpha
\end{align}
$$

    Note that $1-\alpha$ is the confidence interval, and a 95\% confidence interval means that $\alpha = 0.05$. Note that the statement, for example, $\mathbf{P}(.3 \leq \theta \leq .5) \geq 0.95$ does not make any sense because there is no random variable. There is nothing random; $\theta$ is a number, so there are no probabilities associated with it. This statement is wrong on a purely syntactic basis. 

* A confidence interval is actually a statement on the method of determining the parameter $\theta$. It means that 95\% of the time, my method of determining $\theta$ will yield an interval that contains the true value of $\theta$. 

* **The general way of going about determining confidence intervals:** Let's say that we have $X_i$ to be i.i.d. sampled from a distribution that has mean $\mu$ and variance $\sigma^2$. We also define the sample mean $M_n = \sum X_i/n$. We will also use the estimator $\hat{\Theta}_n = M_n$. Now, we are interested in ($1-\alpha$ is the confidence interval)

    $$
    \begin{align}
    & \mathbf{P}\left(\dfrac{\lvert S_n - n\mu \lvert }{\sigma\sqrt{n}}<z\right) \leq 1-\alpha \\
    \Rightarrow & \mathbf{P}\left(\dfrac{\lvert nM_n - n\mu \lvert }{\sigma\sqrt{n}}<z\right) \leq 1-\alpha \\
    \Rightarrow & \mathbf{P}\left(\dfrac{\lvert \hat{\Theta}_n - \theta\lvert }{\sigma/\sqrt{n}}<z\right) \leq 1-\alpha
    \end{align}
    $$

    We can rewrite this as

    $$
    \begin{align}
    &\mathbf{P}\left(-z \leq \dfrac{\hat{\Theta}_n - \theta}{\sigma/\sqrt{n}} \leq z\right) \leq 1-\alpha \\
    \Rightarrow &\mathbf{P}\left(\dfrac{\hat{\Theta}_n - \theta}{\sigma/\sqrt{n}} \leq z \right) - \mathbf{P}\left(\dfrac{\hat{\Theta}_n - \theta}{\sigma/\sqrt{n}} \leq -z \right)  \leq 1-\alpha \\
    \Rightarrow & \Phi(z) - \Phi(-z) \leq 1-\alpha \\
    \Rightarrow & \Phi(z) \leq 1 - \dfrac{\alpha}{2}
    \end{align}
    $$

    Therefore, if we want a 95\% confidence interval, $\alpha=0.05$ and hence we want to find the largest $z$ such that $\Phi(z) =  0.975$. It turns out that this value is $z=1.96$. Now going back to the inequality, 

    $$
    \begin{align}
    -z \leq \dfrac{\hat{\Theta}_n - \theta}{\sigma/\sqrt{n}} \leq z
    \end{align}
    $$

    and rewriting this, 

    $$
    \begin{align}
    \hat{\Theta}-\dfrac{z\sigma}{\sqrt{n}} \leq \theta \leq \hat{\Theta}+\dfrac{z\sigma}{\sqrt{n}}
    \end{align}
    $$

    Therefore, the left hand inequality is the quantity $\hat{\Theta}^-$ and the right hand inequality is the quantity $\hat{\Theta}^+$. To be able to numerically calculate these confidence intervals, we need to know the sample mean and variances. If we do not know that, we have to do some additional work. 

* Sometimes, we do not know $\sigma$ beforehand. In that case, we can either use some underlying knowledge (for example, if $X_i$ are Bernoulli, then we know that $\sigma \leq 1/2$. In other cases, when we only have data, we know the sample mean, and this approaches the true mean $\theta$ if $n$ is large from the weak law of large numbers. We can then find the variance by carrying out the sum (or integral) (1/n)$\sum_i (X_i-\theta)^2$, and this value tends to $\sigma^2$ by the weak law of arge numbers (and so does $\hat{\Theta} \to \theta$).

* Now there are two separate approximations while using this scheme to calculate confidence intervals. (1) We use the CLT and (2) estimating $\sigma^2$ using the weak law of large numbers. Both of these assume $n$ is large. In the case that $n$ is small, instead of using $z$ from the standard normal table, we use the so called t-tables. That is, for a 95\% confidence interval, $z$ is no longer 1.96, but something slightly larger, obtained from the t-tables.  

* **Maximum likelihood estimation:** In ML estimation, we are looking for that value of $\theta$ that makes our observed data most likely. That is, 

    $$
    \begin{align}
    \hat{\theta}\_{ML} = \textrm{arg}\max\limits_\theta f_X(x; \theta)
    \end{align} 
    $$

    It is interesting to compare this to MAP estimators in the Bayesian inference approach. In that case, we are trying to maximize the posterior probability

    $$
    \begin{align}
    f_{\Theta\lvert X}(\theta\lvert x) = \dfrac{f_{X\lvert \Theta}(x\lvert \theta)\cdot f_\Theta(\theta)}{f_X(x)}
    \end{align}
    $$

    The denominator is already a constant. If we had assumed a uniform prior for $\Theta$, maximizing $f_{\Theta\lvert X}(\theta\lvert x)$ amounts to maximizing $f_{X\lvert \Theta}(x\lvert \theta)$ and hence MAP estimation and ML estimation are equivalent in this case of a uniform prior. 

## Unit 9: Bernoulli and Poisson processes
In this unit, we look at so called random or stochastic processes that evolve in time. In the discrete case, we look at the Bernoulli process and in the continuous case, we look at the Poisson process. 

### Lecture 21: The Bernoulli process

* In the Bernoulli process, time is slotted, and we perform a trial $X_i$ at time slot $i$. These trials are all independent. This trial can result in a success (a coin flip that gives heads on trial $i$) or arrival (a customer enters the bank at time $i$ with probability $p$. The \textit{sequence} $X_i$ is then a Bernoulli process i.e. the "process" refers to the sequence of independent trials. 

* The nice thing about Bernoulli processes is that the sum of Bernoulli processes is also a Bernoulli process.

* **Requirements for a Bernoulli process:** All the $X_i$s should be independent and all the $X_i$ should be identically distributed, drawn from the same Bernoulli distribution (i.e.) they all have the same parameter $p$. 

* What exactly is a stochastic process and how is it different from other sequences of random variables that we have dealt with so far. The big difference is that here we are dealing with an *infinite* sequence of events, because time marches on to infinity. For example, in the case of a Bernoulli process, the sequence of trials could look like $\\{0,1,1,0,1,0,1,\ldots\\}$. We are doing a single experiment multiple times. Another such sequence could be $\\{1,0,0,1,0,1,1,1,\ldots\\}$. So in a stochastic process, the sample space $\Omega$ is the set of all such sequences $X_i$. Now, because of independence we can write the joint PMF as a product of the marginal PMF. So when we ask questions such as what is $\mathbf{P}(X_i =1 \textrm{ for all } i)$, we look at the inequality

    $$
    \begin{align}
    \mathbf{P}(X_i =1 \textrm{ for all } i) \leq \mathbf{P}(X_1=1, X_2=1, \ldots, X_n=1)
    \end{align}
    $$

    which equals $p^n$, and as $n$ becomes very large, this probability goes to 0.

* The Bernoulli process has the "fresh-start" property; if I come in and watch the process after $N$ traisl are already taken place, then the process $X_{N+1}, \ldots$ is also Bernoulli, \textit{as long as $N$ is causally determined.} What this means is that $N$ is not determined from some prior knowledge or history of the outcomes of the experiment. For example, if I begin watching the coin flips, just before three sucesses are observed. This would determine $N$ non-causally. An example of $N$ causally determined is if I begin watching the traisl \textit{after} the first heads is observed. 

* Let us now consider the following problem: Let's say that I continually toss a coin, and I am interested in the PMF of the $k^{th}$ arrival i.e. the probability that the $k^{th}$ head occurs at time $t$ (more specifically, after $t$ time slots). Now we define the event $Y_k$ as being the time for the $k^{th}$ arrival. Let $Y_k = T_1 + T_2 + \cdots + T_k$, where $T_i$ are interarrival times. This comes from the fresh-start property. Each of these interarrival times themselves are geometric random variables (with parameter $p$) and hence $\E[Y_k] = \E[\sum T_i] = k/p$. We can do this because all the $T_i$s are independent. Similarly, the variance is $var(Y_k) = k(1-p)/p^2$.Now if we want the PMF of $Y_k$ we are asking for the probability that the $k^{th}$ success occurs at time $t$ i.e., we want $p_{Y_k}(t)$. This is the same as the probability of having $k-1$ successes in $t-1$ trials (which is binomial) AND having a success on the $k^{th}$ trial. Because these events are independent, this is the product of the binomial PMF and $p$. 

    $$
    \begin{align}
    p_{Y_k}(t) = \binom{t-1}{k-1}(1-p)^{k-1}p^{t-k}\cdot p
    \end{align}
    $$

    This is called the Pascal PMF with parameter $p$ and degree $k$.

* **Merging of two Bernoulli processes:** We often talk about merging two Bernoulli processes (say with parameter $p$ and $q$). i.e. we have a random variable $Z_t = P_t + Q_t$. This either takes a value 1 if either $P_t$ or $Q_t$ is 1 else 0. i.e. It is one minus the probability that both $P_t$ and $Q_t$ are zero. Therefore the merged process is also a Bernoulli process with parameter $1-(1-p)(1-q)$. You also need to check that the $Z_t$s are all independent. Can you show this?

* **Splitting of two Bernoulli processes:** Alternatively, we can think about splitting two Bernoulli processes i.e. let's say we have a Bernoulli process $B_i$ with parameter $p$, and that each time we have an arrival, with probability $q$, we send that arrival to one stream (i.e. $X_i =1$)  and with probability $1-q$ we send the process to the other stream (i.e. $Y_i =1$. Now, are these two streams also Bernoulli processes? First we check independence - this can be done intuitively, and it is indeed true that these are independent. Next, we check the probability that one of the streams takes value 1. 

    $$
    \begin{align}
    \mathbf{P}(X_i=1) = \mathbf{P}(X_i=1\lvert B_i=1)\cdot \mathbf{P}(B_i=1) = qp
    \end{align}
    $$

    Therefore, this stream is a Bernoulli process with parameter $pq$. It can also be shown that the other stream is a Bernoulli process with parameter $p(1-q)$. These two streams are \textbf{not} independent! Infomation about one instantly gives us information about the other stream i.e. if one of the steam is a 1, we know that we did not send the value to the other stream and is hence a zero. 

* **Example problem:** For each exam, Ariadne studies with probability 1/2 and does not study with probability 1/2, independently of any other exams. On any exam for which she has not studied, she still has a 0.2  probability of passing, independently of whatever happens on other exams. What is the expected number of total exams taken until she has had 3 exams for which she did not study but which she still passed?

*Solution:* The probability of her not studying and then passing can be thought of as splitting with probability 0.2 a Bernoulli process with parameter 1/2. Therefore, we know that this is a Bernoulli process with parameter 0.5 x 0.2 = 0.1. Now, we have the problem of determining the time of the 3rd arrival for a Bernoulli process with parameter 0.1.

    $$
    \begin{align}
    \E[T_1 +T_2 + T_3] = 3\E[T_i] = 3/.1 = 30
    \end{align}
    $$

    because each $T_i$ is geometric with parameter 0.1 (and hence expected value 10). 

* **The Poisson approximation for the Bernoulli process:** Consider a Bernoulli process where we are interested in the probability of having $k$ successes in $n$ trials, where the number of trials is very large and the probability of a success is very small but their product is a constant i.e. $\lambda=np$. Then the PMF of this case can be calculated as 

    $$
    \begin{align}
    \lim_{n \to \infty} p_S(k) = \lim_{n \to \infty}  \binom{n}{k}p^k(1-p)^{n-k}
    \end{align}
    $$

    which when we take the appropriate limit turns out to be

    $$
    \begin{align}
    \lim_{n \to \infty} p_S(k) = \dfrac{\lambda^k}{k!}e^{-\lambda}
    \end{align}
    $$

    This is called the Poisson approximation to the Bernoulli process. An example is the probability that we will have $k$ earthquakes in $n$ time units, where the probability $p$ of each individual earthquake is very small but $n$ is very large. 

### The Poisson process

* **The definition of a Poisson process:** A Poisson process is just an extension of the Bernoulli process when time is continuous rather than discrete. We are interested in finding the probabilities of having $k$ arrivals in time intervals, rather than time slots as in the Bernoulli process. We have to make the same two assumptions: (1) Independence property i.e. what happens in some particular time interval does not influence what happens in other time intervals. The second assumption is (2) Time homogeneity i.e. the probability of having $k$ arrivals in a time interval of length $\tau$ is the same no matter which interval of length $\tau$ is chosen. We talk of the quantity $\mathbf{P}(k,\tau$). 

* With just the definition above, there is still a problem because it allows for multiple arrivals at any given instant of time. For this, we introduce another condition namely the \textit{very small interval probability} (very small $\delta$):

$$
\begin{align}
\mathbf{P}(k, \tau) \approx
\begin{cases}
0 &, k>1 \\
\lambda\delta &, k=1 \\
1-\lambda\delta &, k=0
\end{cases}
\end{align}
$$

$\lambda$ here is to be interpreted as an arrival rate. 

* The Poisson process applied whenever we have events or arrivals that are somewhat rare that are completely uncoordinated with each other and arrivals happen at totally random times. One of the first applications was deaths by horse kicks in the Prussian army. Another example is radioactive decay. The particle emissions are completely uncoordinated, and each emission itself is pretty rare. What other examples can you think of?

* The PMF of Poisson process can be derived as follows: with some intuitive arguments involving the the very small probability criteria above, we can show that $\mathbf{P}$($k$ arrivals in a Poisson process) $ \approx \mathbf{P}$(having arrivals in $k$ slots in a Bernoulli process). Therefore, we can use the PMF for $k$ arrivals in $n$ slots to find the PMF of the Poisson process. Now we just have to find the equivalent parameter. We initially have to divide the period $\tau$ into $n$ time slots using $n = \tau/\delta$. From the very small probability criteria, $p=\lambda\delta$ and hence $np = \lambda\tau$. Therefore, we replace $\lambda$ with $\lambda\tau$ in $p_S(k)$ and take appropriate limits, and we have for the Poisson process that:

    $$
    \begin{align}
    P(k,\tau) = \dfrac{(\lambda\tau)^k}{k!}e^{-\lambda\tau}; k=0, 1, \ldots
    \end{align}
    $$

    and this is a Poisson process with parameter $\lambda\tau$. Note that $P(k,\tau)$ is just alternate notation for $\mathbf{P}(N_\tau =k)$. Here, $N_\tau$ denotes the number of arrivals in time interval $\tau$.

* It can be shown that the expected value and variance of the Poisson distribution is $\lambda\tau$. 

* **Time for the first arrival:** We want to find the PDF of the time for the first arrival (PDF because time is continuous and this is a continuous random variable). We do this by finding the CDF and then differentiating. The CDF

    $$
    \begin{align}
    F_{T_1}(t) & = \mathbf{P}(T_1 \leq t) \\
    & = 1 - \mathbf{P}(T_1 \geq t)
    \end{align}
    $4

    which is the same as having zero arrivals in time $t$, for which we can use the PMF of the Poisson distribution and this equals $P(0,t) = 1-e^{\lambda t}$. Differentiating this we get,

    $$
    \begin{align}
    f_{T_1}(t) = \lambda e^{-\lambda t}; t \geq 0
    \end{align}
    $$

    This is just an exponential random variable. Therefore the waiting time for the first arrival in a Poisson process is just an exponential random variable. And by memorylessness (or fresh-start), if I enter the room immediately after an arrival has occurred, the time for the next arrival is an exponential random variable. 

* **Time for $k^{th}$ arrival:** The time for the $k^{th}$ arrival $Y_k$ can be derived again from a CDF argument, which can be differentiated to give the PDF (time is continuous and is hence described by a PDF and not a CDF). The CDF is

    $$
    \begin{align}
    F_{Y_k}(y) = \mathbf{P}(Y_k \leq y)
    \end{align}
    $$

    Now the probability that $Y_k \leq y$ is the same as the probability that at least $k$ arrivals have occurred in time $y$, i.e. $k$ or $k+1, \ldots$ arrivals have occured in time y and hence the CDF is

    $$
    \begin{align}
    F_{Y_k}(y) = \sum\limits_{k}^\infty P(k,y)
    \end{align}
    $$

    which can then be differentiated to obtain the PDF. However this is algebraically messy and there is a more intuitive, instructive derivation. The PDF states that

    $$
    \begin{align}
    f_{Y_k}(y)\delta & = \mathbf{P}(y \leq Y_k \leq y+\delta) \\
    & \approx \mathbf{P}(k-1,y) \cdot \lambda\delta
    \end{align}
    $$

    The second equation essentially says that the probability that the $k$-th arrival occurs in the interval $[y, y+\delta]$ is that there are $k-1$ arrivals in time $y$ and then one arrival in a time interval $\delta$. The latter occurs with probability $\lambda\delta$. The former is actually more subtle --- it should also include the case of $k-2$ arrivals in time $y$ and 2 arrivals in time $\delta$. But these higher order probabilities are negligible as $\delta \to 0$. Therefore,

    $$
    \begin{align}
    f_{Y_k}(y) & = \mathbf{P}(k-1,y) \cdot \lambda
    \end{align}
    $$

    and after algebra, we get

    $$
    \begin{align}
    f_{Y_k}(y) = \dfrac{\lambda^k y^{k-1} e^{-\lambda y}}{(k-1)!}; y>0
    \end{align}
    $$

    This distribution is called the \textit{Erlang distribution of order $k$}. There is a different distribution for each $k$, higher the $k$, the broader and more right-shifted the distribution gets, which makes sense intuitively. 


### Lecture 23: More on the Poisson process

* The sum of two Poisson random variables with rates $\lambda$ and $\mu$ respectively is also a Poisson processes with rate $\lambda + \mu$. 

* It can be shown by drawing out a table of possibilities that if we have two Poisson processes merging, the probability that a particular arrival came from the first Poisson process is $\lambda_1/(\lambda_1 + \lambda_2)$, and hence the probability that the arrival came from the other process is $\lambda_1/(\lambda_1 + \lambda_2)$. Now we can also answer questions such as ``if we have ten arrivals, what is the probability that 4 of those arrivals came from process 1?". Each of these arrivals can be trated as Bernoulli trials and hence we can use binomial probabilities to calculate these quantities with $p = \lambda_1/(\lambda_1 + \lambda_2)$.

* Lots of intuitive ways of solving Poisson process problem is by splitting or merging independent processes in lever ways. When the algebra gets too hard, you're probably missing a more intuitive way of solving the process (generally). 

* When splitting and merging and comparing, when we are interested in calculating probabilities involving $k$ arrivals coming from a particular stream, the time of observation doesn't enter into the calculations. We can use simple geometric or binomial probabilities. Given that there were $n$ total arrivals, what is the probability that $k$ of them were from stream 1? Alternatively, what is the probability that there are a string of $k$ arrivals from stream 1 until the first arrival of stream 2. These involve simple (discrete) binomial or geometric calculations. \textit{Time is continuous but arrivals are discrete.}

* See video 9 in lecture 23 (The time until the first (or last) lightbulb burns out). This is a beautiful example on using the intuition of merging Poisson processes and thinking in terms of first arrival times and fresh-start properties to calculate the expected value of a quantity that would otherwise be very hard to find. 

* **Example of the usefulness of merging:** Let $X, Y$, and $Z$ be three independent exponential random variables with parameters $\lambda, \mu$, and $\nu$ respectively. What is the probability that $X<Y<Z$?

Because, interarrival times for each r.v. is an exponential, we can think of the process as being a Poisson process and the PDF for each r.v. is given by the interarrival time expression. Therefore, we now have three independent Poisson processes. We can merge these processes and then forget about time intervals of obervation etc. and just look at discrete probabilities of arrivals. For $X<Y<Z$, we want the first arrival to be $X$ and the second arrival to be $Y$. The probability of the first arrival being $X$ is $\lambda/(\lambda+\mu+\nu)$ and the probability of the second arrival is $\mu/(\mu+\nu)$ and hence

$$
\begin{align}
\mathbf{P}(X<Y<Z) = \dfrac{\lambda}{\lambda+\mu+\nu}\cdot \dfrac{\mu}{\mu+\nu}
\end{align}
$$

* **Splitting of a Poisson process:** We can decide to split a Poisson process into two different streams, and send a particular arrival to stream 1 or stream 2 based on a coin flip with bias $q$. We can verify the time independence by reasoning that nothing that happens in a particular time interval influences another time interval due to the independence of the underlying Poisson process. Second, we can also look at very small interval probabilities and reason that if the time interval goes to zero, we are only going to have 0 or 1 arrival. Therefore, the split streams are also Poisson processes with parameter $\lambda q$. However, something very counter-intuitive is that the *two split streams also also independent.* Unlike in the Bernoulli process where the two split streams are disjoint and hence not independent. 

* The *random incidence paradox* is easily resolved by thinking about backwards Poisson processes. Can you reason about this? Alternatively, think about the memorylessness property of exponential random variables. 

* The random incidence paradox also happens in non-Poisson processes. For example, consider a process where interarrival times are equally likely to be 5 mins or 10 mins. In the this case, the expected value of the interarrival time is expected to be 7.5 mins. However, just the fact that I show up influences what I measure. Because the 10 min intervals run twice as long as the 5 min interval, I am twice as likely to fall in a 10 min interval than in a five minute interval. Therefore, the interarrival time I would measure is (1/3)\*5 + (2/3)\*10 = 8.33 minutes.  

## Unit 10: Markov chains

* A Markov process is a dynamical process whose future evolution depends only on its current state. i.e. conditional on the current state, it's future evolution and past history are independent. It has various wide-ranging application for making predictions on the future evolution of systems. In this course, we only look at discrete time Markov processes to understand all the concepts. But with more technical details, this can be extended to continuous time Markov processes.

### Finite state Markov chains

* Markov chains are more powerful that Bernoulli and Poisson processes because in some sense, we can allow the future of a process depend on its past, rather than them being completely independent. We do this linking between the past and the future with the notion of a \textit{state}, which is described by a probability distribution


{% comment %}

Topics for further practice and review

* Solve all problem sets again; understand all solutions.
* Buffon's needle
* Further practice with complicated cases of Bayes Theorem and Bayesian inference.
* Manipulations of liner combinations of independent normal random variables. How do you find the pdf for the linear combination, what are the mean and variances, what are the expected values?

{% endcomment %}