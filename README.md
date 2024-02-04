# Molecular Information Theory

## What's the goal of this project?
They say a picture is worth a thousand words.  But how many words is a molecule worth?  How much information does a molecule contain?

Our aim is to use information theory to better understand molecular structure.  

## What's information theory?
Imagine you wake up and you know exactly what's going to happen in your day.  Since there's no uncertainty, then your day has no information.  But if you wake up and you have no idea what's going to happen, then your day contains a lot of information.

Information theory quantifies information as the amount of uncertainty.  We can calculate this through <ins>Shannon entropy</ins>:

$$ H(p) = -\sum_{x}p(x)\ \log p(x) $$

[Information theory](https://en.wikipedia.org/wiki/Information_theory) was developed by Claude Shannon in the 1940s.  Today, technologies like the Internet, telecommunication, and CDs would not have been possible without information theory.

## What have you found so far?
### 1) Calculate Shannon entropy (information content) of molecular structure
The information content of molecular structure can be quantified using Shannon entropy.  But Shannon entropy is calculated with respect to a distribution.  So how do we treat conformers as a distribution?  Use their electron density.

<br />

$$ H=-\int_{\vec{r}}p(\vec{r})\ \ln p(\vec{r})\ d\vec{r} $$

<br />

### 2) Information of molecular structure provides a description of electron distribution
We can track how the information of H<sub>2</sub> changes as we vary the bond length:![H2 plot](https://github.com/gomesgroup/shannon_entropy/blob/main/img/h2.png)

The minimum and maximum correspond to the maximum compression and spread of electron density, respectively.  We can confirm this by observing the correspondences between maximum spread of electron density and the equilibrium angle of CH<sub>4</sub>: 

<br />

![CH4 plot](https://github.com/gomesgroup/shannon_entropy/blob/main/img/ch4.png)



### 3) Information scales logarithmically with resolution

An interesting observation: as we vary the resolution of the electron density grid, the information content grows logarithmically.  This is observed across multiple molecules:

![resolution plot](https://github.com/gomesgroup/shannon_entropy/blob/main/img/resolution.png)
