# adaptable experimantalist

- The adaptable experimantalist is a meta sampling method that tries to optimize the use of exesting sampling methods to the current state of an optimization process. It does so by adapting the weights of provided sampling methods, which are projected into an arbitrary samplers space that corresponds to a meta score function.

- we hope that such samplers spaces could yield fitting sampling approaches to various stages of an optimization process, hopefully escaping the downsides of a one-size-fits-all approaches.  
The idea could be further generelized in many aspects.

- The adaptable experimantalist expects a list of sampling methods, their corresponding projection into a samplers space, a list of models from previous optimization steps, an arbitrary meta score function, Tempreture variable, beside usual parameters like the conditions pool and the number of wanted samples.  

    By default, the adaptable experimantalist uses a surprisal score function based on the jensen shannon divergence between the current and previous models. this score is then used to adapt the weights of the provided sampling methods.Here assuming a one-dimensional samplers space, where the lower end samplers are better in lower surprisal scores and vice versa.

- the adaptable experimantalist is a meta sampling method. By default, it is set to use a set of Novelity, falsification, model disagreement and confirmation methods, as this fits well to the default meta score function.
We went with a meta method as we acknowledged that perhabs it is not the lack of sampling methods that is holding the sampling efficiency back, but rather a lack of adaptability to the current state of the optimization process, as we think sampling is highly tied to the naturally unkown approximity of the underlying truth, we aim to assess such approximity and adapt the sampling methods accordingly.


