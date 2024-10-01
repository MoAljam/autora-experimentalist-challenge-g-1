# Adaptable Experimentalist

- The Adaptable Experimentalist is a meta-sampling method that tries to optimize the use of existing sampling methods according to the current state of an optimization process. It does so by adapting the weights of the provided sampling methods, which are projected into an arbitrary sampler space that corresponds to a meta score function.

- We hope that such sampler spaces could yield appropriate sampling approaches for various stages of an optimization process, hopefully escaping the downsides of a one-size-fits-all approach.  
The idea could be further generalized in many aspects.

- The Adaptable Experimentalist expects a list of sampling methods, their corresponding projections into a sampler space, a list of models from previous optimization steps, an arbitrary meta score function, a temperature variable, besides usual parameters like the conditions pool and the number of desired samples.

    By default, the Adaptable Experimentalist uses a surprisal score function based on the Jensen-Shannon divergence between the current and previous models. This score is then used to adapt the weights of the provided sampling methods, assuming a one-dimensional sampler space where the lower-end samplers are better at lower surprisal scores, and vice versa.

- The Adaptable Experimentalist is a meta-sampling method. By default, it is set to use a set of novelty, falsification, model disagreement, and confirmation methods, as this fits well with the default meta score function.
  
  We went with a meta method as we acknowledged that perhaps it is not the lack of sampling methods that is holding the sampling efficiency back, but rather a lack of adaptability to the current state of the optimization process. As we think sampling is highly tied to the naturally unknown proximity to the underlying truth, we aim to assess such proximity and adapt the sampling methods accordingly.