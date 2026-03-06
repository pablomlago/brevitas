
Learned Round
=============

Learned Round is a **post-training quantization (PTQ)** technique that improves quantization quality by **learning per-weight
rounding decisions**, instead of relying on fixed round-to-nearest (RTN). It unifies methods such as **AdaRound** [1]_ and
**SignRound** [2]_ under a single, configurable framework integrated into Brevitas’ PTQ pipelines.

.. contents:: Table of Contents
   :local:
   :depth: 3


About the Algorithm
-------------------

Motivation
~~~~~~~~~~

Quantization mappings generally **require a rounding operator**, for which **round-to-nearest (RTN)** is the standard choice.

For example, in symmetric integer quantization the mapping is often written as:

.. math::

    \mathcal{Q}(W) := s \cdot \left(
    \text{clip}\left(
        \left\lceil \frac{W}{s} \right\rfloor + z,
        \min \mathcal{A}, \max \mathcal{A}
    \right) - z
    \right).

RTN is optimal when minimizing **weight reconstruction error**,

.. math::

    \lVert W - \mathcal{Q}(W) \rVert_2,

but this optimality **does not hold when considering the layer output reconstruction loss** (or, depending on the pipeline,
a block-wise variant), e.g.:

.. math::

    \lVert XW - X\mathcal{Q}(W) \rVert_2,

which is commonly used as a proxy for downstream accuracy degradation during PTQ.

This observation motivates **learned rounding**, where each weight is allowed to round **up or down** in a data-driven way.

Learned rounding as optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Methods such as **AdaRound** [1]_ and **SignRound** [2]_ formulate rounding as a **binary optimization problem**, choosing
between the floor or ceiling of the quantization grid for each weight. Although the discrete problem is NP-hard, it can be
relaxed into a **continuous optimization** by introducing learnable parameters inside the rounding operator and optimizing
them using calibration data.

Unlike greedy solvers such as **GPTQ** [3]_ and **Qronos** [4]_, which typically solve closed-form layer-wise objectives
sequentially, learned rounding methods:

- jointly optimize rounding decisions (per layer / per block, depending on the pipeline),
- use gradient-based optimization over calibration data,
- restrict the search space to a limited subset of quantization grid points.

This can improve robustness and reduce overfitting to calibration data, at the cost of additional compute.

Learned Round in Brevitas
~~~~~~~~~~~~~~~~~~~~~~~~~

In Brevitas, these techniques are unified under the name **Learned Round**, providing:

- a common abstraction for learned rounding,
- flexible choices of rounding parametrization and optimization strategy,
- seamless integration with existing PTQ pipelines.

Learned Round is compatible with **all quantized data types currently available in Brevitas**, including:

- integer quantization (e.g. INT2/INT4/INT8),
- weight-only, weight-and-activation, and KV-cache quantization,
- advanced formats such as **MXFP4**.

It is also composable with outlier suppression and rotation-based methods such as **QuaRot** [5]_, **SpinQuant** [6]_,
and **MagR** [7]_.


Rounding Parameterizations
--------------------------

Learned Round reformulates rounding as:

.. math::

    \text{round}(w) = \lfloor w \rfloor + \Delta(w; p),

where :math:`p` is a learnable parameter controlling the rounding direction.

Brevitas provides multiple parameterizations implemented in
``brevitas/core/function_wrapper/learned_round.py``. Two commonly used choices are:

- **Sigmoid** (AdaRound-style):

  .. math::

     \text{round}(p; w, T) = \lfloor w \rfloor + \sigma(p / T)

- **Identity** (SignRound-style):

  .. math::

     \text{round}(p; w) =
     \left\lfloor w + \text{clip}(p, -0.5, 0.5) \right\rceil


Implementation Overview
-----------------------

Learned Round is available through Brevitas’ PTQ pipelines, including the LLM and ImageNet entrypoints.

At a high level, the Learned Round workflow:

1. collects calibration data,
2. freezes the quantization grid,
3. optimizes rounding parameters (and optionally scales),
4. fixes rounding decisions for inference.

All orchestration is handled by ``LearnedRoundOptimizer``.

LearnedRoundOptimizer
~~~~~~~~~~~~~~~~~~~~~

``LearnedRoundOptimizer`` wires together:

- a learned rounding implementation (e.g. Identity or Sigmoid),
- a reconstruction loss (e.g. MSE or regularized MSE),
- optimizers and learning-rate schedulers,
- PTQ configuration (iterations, batch size, AMP dtype, etc.).

For example, a configuration equivalent to **SignRound** [2]_ (without scale learning) looks like:

.. code-block:: python
   :caption: Example LearnedRoundOptimizer instantiation

    learned_round_optimizer = LearnedRoundOptimizer(
        learned_round=LearnedRoundImplType.IDENTITY,
        learned_round_loss_class=MSELoss,
        optimizer_class=SignSGD,
        lr_scheduler_class=LinearLR,
        batch_size=8,
        iters=200,
        learn_scale=False,
        use_best_model=True,
        amp_dtype=torch.float16,
        loss_scaling_factor=1000.,
        optimizer_kwargs={"lr": 5e-3},
        lr_scheduler_kwargs={"start_factor": 1.0, "end_factor": 0.0},
    )

Entrypoint integration (important)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you are using Brevitas’ provided entrypoints (e.g., the LLM PTQ entrypoint or ImageNet PTQ scripts):

- ✅ You **do not need** to implement caches, block forward functions, or block extraction logic.
- ✅ The required plumbing is already provided and configured internally.

The lower-level abstractions (cache objects, block forward hooks, block lists, etc.) are primarily relevant if you are
building a **custom PTQ pipeline** outside the supported entrypoints.

Example instantiations for the LLM and ImageNet entrypoints can be found at:

- ``brevitas_examples/llm/llm_quant/learned_round_utils.py``
- ``brevitas_examples/imagenet_classification/ptq/learned_round_utils.py``


Getting Started
---------------

Install Brevitas with LLM support:

.. code-block:: shell

   pip install brevitas[llm,export]

If you are evaluating with LightEval:

.. code-block:: shell

   pip install lighteval

Then enable Learned Round in your chosen PTQ configuration / entrypoint utilities (see the files linked above).


Results
-------

Learned Round has been evaluated across multiple scenarios including weight-only and weight-and-activation PTQ, and in
combination with outlier suppression techniques. In general, Learned Round:

- consistently outperforms RTN,
- matches or improves upon SignRound-style baselines depending on the configuration,
- is competitive with or complementary to greedy solvers like GPTQ and Qronos,
- remains effective when composed with rotation/outlier-suppression methods.

To keep this page concise (and avoid large inline YAML dumps / tables), **full experiment tables and YAML configurations
should live in the papers/examples section**, rather than embedded directly here.

Recommended layout:

- ``brevitas_examples/papers/learned_round/README.md`` (tables + narrative)
- ``brevitas_examples/papers/learned_round/configs/`` (YAML configs)


Extending Learned Round
-----------------------

This section is intended for advanced users who want to add a custom learned rounding parametrization.

Recommended workflow:

1. Subclass ``brevitas.jit.ScriptModule``
2. Implement ``forward`` and ``round_forward`` similarly to existing implementations in
   ``brevitas/core/function_wrapper/learned_round.py``
3. Register the implementation in:
   - ``LearnedRoundImplType`` (``brevitas/inject/enum.py``)
   - ``SolveTensorQuantFloatToIntImplFromEnum`` (``brevitas/quant/solver/common.py``)

Minimal example:

.. code-block:: python
   :caption: Custom learned round parametrization

    class LearnedRoundCustom(brevitas.jit.ScriptModule):
        """
        Custom learned round parametrization.
        """
        @brevitas.jit.script_method
        def forward(self, p: torch.Tensor) -> torch.Tensor:
            ...

        @brevitas.jit.script_method
        def round_forward(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
            ...

Results
==============

To demonstrate the effectivenes and flexibility of the Learned Round implementation in Brevitas,
its performance was compared against the Sign Round [2]_ for weight-only quantization,
and against GPTQ and Qronos for the rest of scenarios.

In comparison with Sign Round [2]_, Signed SGD was also used in these experiments,
but the number of iterations and the learning rate were decoupled, thus requiring the clipping operation in ``LearnedRoundIdentity``.
Moreover, the SGD optimizer was used for learning the scales, and these are parametrized directly, instead of learning the weight clipping,
while in Sign Round [2]_ the authors use Sign SGD to learn the weight clipping, in the same fashion as OmniQuant [8]_.

Experiments were conducted on **Llama 3.2** and **Qwen 2.5** base models, sourced from **Huggingface**, using **WikiText2** for validation.
To assess generalization, **LightEval** was used across five zero-shot reasoning tasks, reporting the normalized average accuracy for these:

- ARC (challenge and easy)
- HellaSwag
- PIQA
- Winogrande

Weight-only quantization of `Llama 3.2` and `Qwen 2.5` foundation models
--------------------------------------------------------------------------

The quantization configuration used is:

.. code-block:: yaml

    scaling_min_val:
    - 0.0001
    weight_bit_width:
    - 2
    - 4
    weight_group_dim:
    - null
    weight_group_size:
    - 128
    weight_param_method:
    - stats
    weight_quant_format:
    - int
    weight_quant_granularity:
    - per_channel
    - per_group
    weight_quant_type:
    - sym
    weight_scale_precision:
    - float_scale

The results for `Llama 3.2` are summarized in the following table:

+------------+-----------+------------------+-------------+-------------+-------------+------------+------------+------------+-------------+-------------+-------------+------------+------------+------------+-------------+-------------+-------------+------------+------------+------------+
|                                           |                                     **W2g128**                                 |                                       **W4**                                   |                                     **W4g128**                                 |
+------------+-----------+------------------+-------------+-------------+-------------+------------+------------+------------+-------------+-------------+-------------+------------+------------+------------+-------------+-------------+-------------+------------+------------+------------+
|                                           |               WikiText2 ↓               |               0-shot ↑               |               WikiText2 ↓               |               0-shot ↑               |               WikiText2 ↓               |               0-shot ↑               |
+------------+-----------+------------------+-------------+-------------+-------------+------------+------------+------------+-------------+-------------+-------------+------------+------------+------------+-------------+-------------+-------------+------------+------------+------------+
| Model      | Stage 1   | Stage 2          |  1B         |  3B         |  8B         |   1B       |   3B       |   8B       |  1B         |  3B         |  8B         |   1B       |   3B       |   8B       |  1B         |  3B         |  8B         |   1B       |   3B       |   8B       |
+------------+-----------+------------------+-------------+-------------+-------------+------------+------------+------------+-------------+-------------+-------------+------------+------------+------------+-------------+-------------+-------------+------------+------------+------------+
| Llama-3.2  | BF16      |                  |  8.9        |  7.2        |  5.9        | 56.2       | 63.6       | 69.1       |  8.9        |  7.2        |  5.9        | 56.2       | 63.6       | 69.1       |  8.9        |  7.2        |  5.9        | 56.2       | 63.6       | 69.1       |
+            +-----------+------------------+-------------+-------------+-------------+------------+------------+------------+-------------+-------------+-------------+------------+------------+------------+-------------+-------------+-------------+------------+------------+------------+
|            | -         | RTN              | 92672.00    | 11776.00    | 38656.00    | 35.06      | 35.54      | 35.57      | 23.12       | 9.81        | 7.88        | 48.50      | 58.72      | 65.23      | 11.06       | 7.75        | 6.38        | 52.83      | 61.57      | 68.31      |
+            +           +------------------+-------------+-------------+-------------+------------+------------+------------+-------------+-------------+-------------+------------+------------+------------+-------------+-------------+-------------+------------+------------+------------+
|            |           | GPTQ             | 179.00      | 33.00       | 25.38       | 36.78      | 41.08      | 43.60      | 11.06       | 8.12        | 6.78        | 53.40      | 61.48      | 66.52      | 9.81        | 7.50        | 6.22        | 54.93      | 62.49      | 68.27      |
+            +           +------------------+-------------+-------------+-------------+------------+------------+------------+-------------+-------------+-------------+------------+------------+------------+-------------+-------------+-------------+------------+------------+------------+
|            |           | Qronos           | 60.00       | 21.00       | 16.12       | 38.84      | 45.68      | 50.20      | 10.75       | 7.88        | 6.62        | 53.83      | 62.00      | 67.18      | 9.62        | 7.38        | 6.19        | 55.23      | 62.82      | 68.31      |
+            +           +------------------+-------------+-------------+-------------+------------+------------+------------+-------------+-------------+-------------+------------+------------+------------+-------------+-------------+-------------+------------+------------+------------+
|            |           | Sign Round       | 17151.00    | 36352.00    | 6304.00     | 41.71      | 51.06      | 55.21      | 10.12       | 13.38       | 10.75       | 54.73      | 62.74      | 68.20      | 9.62        | 7.38        | 6.12        | 55.23      | 63.17      | 68.37      |
+            +           +------------------+-------------+-------------+-------------+------------+------------+------------+-------------+-------------+-------------+------------+------------+------------+-------------+-------------+-------------+------------+------------+------------+
|            |           | Learned Round    | 41.67       | 18.18       | 14.13       | 43.66      | 48.76      | 55.47      | 10.44       | 8.11        | 6.48        | 54.12      | 62.62      | 67.09      | 9.57        | 7.44        | 6.12        | 55.23      | 63.08      | 68.20      |
+            +-----------+------------------+-------------+-------------+-------------+------------+------------+------------+-------------+-------------+-------------+------------+------------+------------+-------------+-------------+-------------+------------+------------+------------+
|            | HIP       | RTN              | 143360.00   | 16128.00    | 4928.00     | 34.79      | 35.06      | 35.40      | 12.94       | 9.06        | 7.09        | 50.85      | 59.54      | 66.64      | 10.94       | 8.00        | 6.47        | 53.50      | 61.72      | 67.91      |
+            +           +------------------+-------------+-------------+-------------+------------+------------+------------+-------------+-------------+-------------+------------+------------+------------+-------------+-------------+-------------+------------+------------+------------+
|            |           | GPTQ             | 131.00      | 27.00       | 18.62       | 37.35      | 43.10      | 48.70      | 10.25       | 7.75        | 6.47        | 54.27      | 62.37      | 67.00      | 9.62        | 7.50        | 6.19        | 55.22      | 63.15      | 68.16      |
+            +           +------------------+-------------+-------------+-------------+------------+------------+------------+-------------+-------------+-------------+------------+------------+------------+-------------+-------------+-------------+------------+------------+------------+
|            |           | Qronos           | 77.00       | 35.25       | 20.75       | 38.38      | 41.31      | 46.14      | 10.56       | 8.12        | 6.62        | 52.94      | 61.49      | 66.53      | 9.94        | 7.62        | 6.28        | 55.02      | 62.89      | 68.35      |
+            +           +------------------+-------------+-------------+-------------+------------+------------+------------+-------------+-------------+-------------+------------+------------+------------+-------------+-------------+-------------+------------+------------+------------+
|            |           | Learned Round    | 32.53       | 17.64       | 13.09       | 43.97      | 50.57      | 36.05      | 9.74        | 7.65        | 6.31        | 55.37      | 62.82      | 67.98      | 9.40        | 7.42        | 6.09        | 55.97      | 63.29      | 68.58      |
+            +-----------+------------------+-------------+-------------+-------------+------------+------------+------------+-------------+-------------+-------------+------------+------------+------------+-------------+-------------+-------------+------------+------------+------------+
|            | MagR      | RTN              | 20736.00    | 16128.00    | 5568.00     | 35.81      | 35.44      | 35.57      | 13.19       | 9.06        | 7.09        | 50.84      | 54.51      | 65.12      | 12.19       | 8.50        | 6.78        | 51.84      | 56.34      | 65.08      |
+            +           +------------------+-------------+-------------+-------------+------------+------------+------------+-------------+-------------+-------------+------------+------------+------------+-------------+-------------+-------------+------------+------------+------------+
|            |           | GPTQ             | 96.00       | 34.25       | 25.38       | 37.10      | 38.90      | 42.68      | 11.25       | 8.38        | 6.69        | 53.35      | 57.26      | 66.64      | 10.75       | 8.12        | 6.53        | 53.67      | 58.81      | 66.80      |
+            +           +------------------+-------------+-------------+-------------+------------+------------+------------+-------------+-------------+-------------+------------+------------+------------+-------------+-------------+-------------+------------+------------+------------+
|            |           | Qronos           | 43.75       | 21.75       | 18.25       | 40.23      | 45.90      | 51.31      | 10.56       | 7.75        | 6.41        | 54.39      | 61.61      | 67.44      | 10.25       | 7.62        | 6.28        | 54.83      | 61.50      | 67.85      |
+            +           +------------------+-------------+-------------+-------------+------------+------------+------------+-------------+-------------+-------------+------------+------------+------------+-------------+-------------+-------------+------------+------------+------------+
|            |           | Learned Round    | 34.16       | 17.98       |             | 43.83      | 49.63      |            | 10.07       | 8.11        |             | 52.58      | 46.94      |            | 9.89        | 7.91        |             | 54.31      | 56.37      |            |
+------------+-----------+------------------+-------------+-------------+-------------+------------+------------+------------+-------------+-------------+-------------+------------+------------+------------+-------------+-------------+-------------+------------+------------+------------+

The results for `Qwen 2.5` are summarized in the following table:

+------------+-----------+------------------+-------------+-------------+-------------+------------+------------+------------+-------------+-------------+-------------+------------+------------+------------+-------------+-------------+-------------+------------+------------+------------+
|                                           |                                     **W2g128**                                 |                                       **W4**                                   |                                     **W4g128**                                 |
+------------+-----------+------------------+-------------+-------------+-------------+------------+------------+------------+-------------+-------------+-------------+------------+------------+------------+-------------+-------------+-------------+------------+------------+------------+
|                                           |               WikiText2 ↓               |               0-shot ↑               |               WikiText2 ↓               |               0-shot ↑               |               WikiText2 ↓               |               0-shot ↑               |
+------------+-----------+------------------+-------------+-------------+-------------+------------+------------+------------+-------------+-------------+-------------+------------+------------+------------+-------------+-------------+-------------+------------+------------+------------+
| Model      | Stage 1   | Stage 2          |  1.5B       |  3B         |  7B         |   1.5B     |   3B       |   7B       |  1.5B       |  3B         |  7B         |   1.5B     |   3B       |   7B       |  1.5B       |  3B         |  7B         |   1.5B     |   3B       |   7B       |
+------------+-----------+------------------+-------------+-------------+-------------+------------+------------+------------+-------------+-------------+-------------+------------+------------+------------+-------------+-------------+-------------+------------+------------+------------+
| Qwen 2.5   | BF16      |                  |  8.5        |  7.4        |  6.5        | 60.7       | 64.3       | 67.2       | 8.5         |  7.4        | 6.5         | 60.7       | 64.3       | 67.2       | 8.5         | 7.4         | 6.5         | 60.7       | 64.3       | 67.2       |
+            +-----------+------------------+-------------+-------------+-------------+------------+------------+------------+-------------+-------------+-------------+------------+------------+------------+-------------+-------------+-------------+------------+------------+------------+
|            | -         | RTN              | 152576.00   | 76800.00    | 19456.00    | 35.33      | 34.90      | 35.53      | 12.75       | 6304.00     | 8.50        | 54.52      | 35.56      | 61.49      | 9.50        | 9.06        | 6.78        | 58.47      | 61.37      | 65.61      |
+            +           +------------------+-------------+-------------+-------------+------------+------------+------------+-------------+-------------+-------------+------------+------------+------------+-------------+-------------+-------------+------------+------------+------------+
|            |           | GPTQ             | 38.00       | 23.12       | 12.56       | 39.46      | 41.43      | 52.07      | 9.81        | 8.38        | 7.09        | 56.59      | 62.24      | 64.16      | 8.94        | 7.75        | 6.69        | 59.38      | 62.85      | 66.36      |
+            +           +------------------+-------------+-------------+-------------+------------+------------+------------+-------------+-------------+-------------+------------+------------+------------+-------------+-------------+-------------+------------+------------+------------+
|            |           | Qronos           | 27.50       | 18.62       | 12.19       | 42.57      | 46.41      | 55.23      | 9.50        | 8.25        | 7.06        | 56.42      | 62.41      | 65.33      | 8.94        | 7.75        | 6.69        | 60.14      | 62.47      | 66.75      |
+            +           +------------------+-------------+-------------+-------------+------------+------------+------------+-------------+-------------+-------------+------------+------------+------------+-------------+-------------+-------------+------------+------------+------------+
|            |           | Sign Round       | 26.62       | 18.00       | 11.62       | 46.56      | 50.65      | 59.26      | 9.19        | 8.00        | 6.84        | 58.88      | 62.41      | 65.93      | 8.94        | 7.75        | 6.62        | 60.46      | 63.99      | 66.79      |
+            +           +------------------+-------------+-------------+-------------+------------+------------+------------+-------------+-------------+-------------+------------+------------+------------+-------------+-------------+-------------+------------+------------+------------+
|            |           | Learned Round    | 23.70       | 16.93       | 12.09       | 45.96      | 51.00      | 57.73      | 9.85        | 8.10        | 10.04       | 59.28      | 63.27      | 65.34      | 8.86        | 7.73        | 6.68        | 59.73      | 64.22      | 66.84      |
+            +-----------+------------------+-------------+-------------+-------------+------------+------------+------------+-------------+-------------+-------------+------------+------------+------------+-------------+-------------+-------------+------------+------------+------------+
|            | HIP       | RTN              | 14208.00    | 95420416.00 | 536.00      | 34.83      | 35.06      | 37.65      | 9.94        | 11.81       | 8.00        | 56.84      | 59.66      | 62.94      | 9.31        | 8.25        | 6.78        | 59.73      | 62.51      | 65.94      |
+            +           +------------------+-------------+-------------+-------------+------------+------------+------------+-------------+-------------+-------------+------------+------------+------------+-------------+-------------+-------------+------------+------------+------------+
|            |           | GPTQ             | 23.12       | 15.88       | 10.94       | 43.71      | 45.93      | 52.79      | 9.06        | 7.88        | 6.94        | 59.81      | 63.56      | 65.70      | 8.75        | 7.62        | 6.62        | 59.66      | 63.73      | 66.48      |
+            +           +------------------+-------------+-------------+-------------+------------+------------+------------+-------------+-------------+-------------+------------+------------+------------+-------------+-------------+-------------+------------+------------+------------+
|            |           | Qronos           | 20.38       | 15.19       | 10.75       | 45.99      | 47.02      | 55.92      | 9.06        | 7.88        | 6.94        | 58.69      | 62.78      | 66.19      | 8.75        | 7.75        | 6.62        | 60.29      | 63.16      | 66.87      |
+            +           +------------------+-------------+-------------+-------------+------------+------------+------------+-------------+-------------+-------------+------------+------------+------------+-------------+-------------+-------------+------------+------------+------------+
|            |           | Learned Round    | 18.72       | 13.48       | 11.49       | 46.55      | 52.08      | 57.49      | 11.84       | 7.99        | 7.46        | 52.57      | 64.18      | 65.72      | 8.74        | 7.63        | 6.69        | 59.93      | 64.06      | 66.42      |
+            +-----------+------------------+-------------+-------------+-------------+------------+------------+------------+-------------+-------------+-------------+------------+------------+------------+-------------+-------------+-------------+------------+------------+------------+
|            | MagR      | RTN              | 56320.00    | 68096.00    | 1696.00     | 35.55      | 35.15      | 36.15      | 10.56       | 9.06        | 7.50        | 55.31      | 59.89      | 64.93      | 10.12       | 8.62        | 7.28        | 56.66      | 61.02      | 66.33      |
+            +           +------------------+-------------+-------------+-------------+------------+------------+------------+-------------+-------------+-------------+------------+------------+------------+-------------+-------------+-------------+------------+------------+------------+
|            |           | GPTQ             | 43.75       | 40.00       | 13.81       | 40.42      | 42.54      | 51.59      | 9.94        | 8.38        | 7.34        | 58.19      | 62.56      | 65.39      | 9.62        | 8.25        | 7.16        | 58.04      | 61.89      | 66.12      |
+            +           +------------------+-------------+-------------+-------------+------------+------------+------------+-------------+-------------+-------------+------------+------------+------------+-------------+-------------+-------------+------------+------------+------------+
|            |           | Qronos           | 34.25       | 19.75       | 13.81       | 41.37      | 46.71      | 54.87      | 9.81        | 8.38        | 7.28        | 57.71      | 61.56      | 65.65      | 9.50        | 8.00        | 7.16        | 57.90      | 61.51      | 66.38      |
+            +           +------------------+-------------+-------------+-------------+------------+------------+------------+-------------+-------------+-------------+------------+------------+------------+-------------+-------------+-------------+------------+------------+------------+
|            |           | Learned Round    | 22.49       | 15.79       |             | **46.83**  | 48.03      |            | 10.14       | 8.08        |             | 58.21      | 63.07      |            | 9.03        | 7.80        |             | 59.14      | 63.44      |            |
+------------+-----------+------------------+-------------+-------------+-------------+------------+------------+------------+-------------+-------------+-------------+------------+------------+------------+-------------+-------------+-------------+------------+------------+------------+


Weight and activation quantization of `Llama 3.2` foundation models
--------------------------------------------------------------------

The quantization configuration used is:

.. code-block:: yaml

    scaling_min_val:
    - 0.0001
    weight_bit_width:
    - 4
    weight_group_dim:
    - null
    weight_group_size:
    - null
    weight_param_method:
    - stats
    weight_quant_format:
    - int
    weight_quant_granularity:
    - per_channel
    weight_quant_type:
    - sym
    weight_scale_precision:
    - float_scale
    input_bit_width:
    - 4
    input_group_size:
    - 32
    input_param_method:
    - stats
    input_quant_format:
    - int
    input_quant_granularity:
    - per_row
    input_quant_type:
    - asym
    input_scale_precision:
    - float_scale
    input_scale_type:
    - dynamic

The results for `Llama 3.2` are summarized in the following table:

+------------+-----------+------------------+-------------+-------------+-------------+------------+------------+------------+
|                                           |                                     **W4A4**                                   |
+------------+-----------+------------------+-------------+-------------+-------------+------------+------------+------------+
|                                           |               WikiText2 ↓               |               0-shot ↑               |
+------------+-----------+------------------+-------------+-------------+-------------+------------+------------+------------+
| Model      | Stage 1   | Stage 2          |  1B         |  3B         |  8B         |   1B       |   3B       |   8B       |
+------------+-----------+------------------+-------------+-------------+-------------+------------+------------+------------+
| Llama-3.2  | BF16      |                  |  8.9        |  7.2        |  5.9        | 56.2       | 63.6       | 69.1       |
+            +-----------+------------------+-------------+-------------+-------------+------------+------------+------------+
|            | -         | RTN              | 6304.00     | 22016.00    | 52736.00    | 34.59      | 34.83      | 35.60      |
+            +           +------------------+-------------+-------------+-------------+------------+------------+------------+
|            |           | GPTQ             | 23424.00    | 14208.00    | 23424.00    | 34.38      | 35.48      | 34.32      |
+            +           +------------------+-------------+-------------+-------------+------------+------------+------------+
|            |           | Qronos           | 174.00      | 84.50       | 82.00       | 37.44      | 38.59      | 38.65      |
+            +           +------------------+-------------+-------------+-------------+------------+------------+------------+
|            |           | Learned Round    | 100.67      | 73.80       | 274.88      | 36.10      | 39.03      | 38.15      |
+            +-----------+------------------+-------------+-------------+-------------+------------+------------+------------+
|            | HIP       | RTN              | 18.25       | 10.56       | 8.38        | 45.78      | 55.25      | 61.33      |
+            +           +------------------+-------------+-------------+-------------+------------+------------+------------+
|            |           | GPTQ             | 13.19       | 8.75        | 7.50        | 48.49      | 58.35      | 62.76      |
+            +           +------------------+-------------+-------------+-------------+------------+------------+------------+
|            |           | Qronos           | 13.19       | 9.19        | 7.62        | 48.40      | 58.24      | 62.85      |
+            +           +------------------+-------------+-------------+-------------+------------+------------+------------+
|            |           | Learned Round    | 12.32       | 8.78        | 7.23        | 50.57      | 59.09      | 63.70      |
+            +-----------+------------------+-------------+-------------+-------------+------------+------------+------------+
|            | MagR      | RTN              | 5920.00     | 8096.00     | 24960.00    | 34.94      | 35.03      | 34.75      |
+            +           +------------------+-------------+-------------+-------------+------------+------------+------------+
|            |           | GPTQ             | 12544.00    | 17152.00    | 24960.00    | 35.74      | 35.91      | 35.44      |
+            +           +------------------+-------------+-------------+-------------+------------+------------+------------+
|            |           | Qronos           | 197.00      | 153.00      | 174.00      | 36.74      | 37.65      | 38.05      |
+            +           +------------------+-------------+-------------+-------------+------------+------------+------------+
|            |           | Learned Round    | 103.10      | 82.66       |             | 38.74      | 36.75      |            |
+            +-----------+------------------+-------------+-------------+-------------+------------+------------+------------+
|            | QuaRot    | RTN              | 27.88       | 19.12       | 11.62       | 42.25      | 44.93      | 55.34      |
+            +           +------------------+-------------+-------------+-------------+------------+------------+------------+
|            |           | GPTQ             | 14.69       | 10.12       | 8.00        | 47.54      | 55.34      | 61.58      |
+            +           +------------------+-------------+-------------+-------------+------------+------------+------------+
|            |           | Qronos           | 13.81       | 9.31        | 7.75        | 48.77      | 57.18      | 62.80      |
+            +           +------------------+-------------+-------------+-------------+------------+------------+------------+
|            |           | Learned Round    | 13.65       | 9.88        | 7.86        | 49.26      | 55.22      | 44.84      |
+            +-----------+------------------+-------------+-------------+-------------+------------+------------+------------+
|            | SpinQuant | RTN              | 18.25       | 87.00       | 77.00       | 46.57      | 35.06      | 36.15      |
+            +           +------------------+-------------+-------------+-------------+------------+------------+------------+
|            |           | GPTQ             | 15.38       | 1240.00     | 392.00      | 47.50      | 34.58      | 34.67      |
+            +           +------------------+-------------+-------------+-------------+------------+------------+------------+
|            |           | Qronos           | 14.69       | 368.00      | 286.00      | 47.81      | 34.68      | 35.27      |
+            +           +------------------+-------------+-------------+-------------+------------+------------+------------+
|            |           | Learned Round    | 13.52       | 9.41        | 7.59        | 50.22      | 57.08      | 62.22      |
+------------+-----------+------------------+-------------+-------------+-------------+------------+------------+------------+

MXFP4 weight and activation quantization of `Llama 3.2` foundation models
----------------------------------------------------------------------------

The quantization configuration used is:

.. code-block:: yaml

    scaling_min_val:
    - 0.0001
    weight_bit_width:
    - 4
    weight_group_dim:
    - null
    weight_group_size:
    - 32
    weight_param_method:
    - stats
    weight_quant_format:
    - float_ocp_e2m1
    weight_quant_granularity:
    - per_group
    weight_quant_type:
    - sym
    weight_scale_precision:
    - po2_scale
    scale_rounding_func_type:
    - floor

    input_bit_width:
    - 4
    input_group_size:
    - 32
    input_param_method:
    - stats
    input_quant_format:
    - float_ocp_e2m1
    input_quant_granularity:
    - per_group
    input_quant_type:
    - sym
    input_scale_precision:
    - po2_scale
    input_scale_type:
    - dynamic


The results for `Llama 3.2` are summarized in the following table:

+------------+-----------+------------------+-------------+-------------+-------------+------------+------------+------------+
|                                           |                                     **W4g32A**                                 |
+------------+-----------+------------------+-------------+-------------+-------------+------------+------------+------------+
|                                           |               WikiText2 ↓               |               0-shot ↑               |
+------------+-----------+------------------+-------------+-------------+-------------+------------+------------+------------+
| Model      | Stage 1   | Stage 2          |  1B         |  3B         |  8B         |   1B       |   3B       |   8B       |
+------------+-----------+------------------+-------------+-------------+-------------+------------+------------+------------+
| Llama-3.2  | BF16      |                  |  8.9        |  7.2        |  5.9        | 56.2       | 63.6       | 69.1       |
+            +-----------+------------------+-------------+-------------+-------------+------------+------------+------------+
|            | -         | RTN              | 14.44       | 9.19        | 7.75        | 50.15      | 57.39      | 63.45      |
+            +           +------------------+-------------+-------------+-------------+------------+------------+------------+
|            |           | GPTQ             | 12.38       | 8.62        | 7.16        | 51.80      | 56.95      | 64.68      |
+            +           +------------------+-------------+-------------+-------------+------------+------------+------------+
|            |           | Qronos           | 12.56       | 8.75        | 7.22        | 51.57      | 59.14      | 64.09      |
+            +           +------------------+-------------+-------------+-------------+------------+------------+------------+
|            |           | Learned Round    | 11.78       | 8.49        | 6.97        | 52.78      | 61.01      | 65.21      |
+            +-----------+------------------+-------------+-------------+-------------+------------+------------+------------+
|            | HIP       | RTN              | 13.19       | 8.94        | 7.28        | 50.42      | 59.21      | 65.99      |
+            +           +------------------+-------------+-------------+-------------+------------+------------+------------+
|            |           | GPTQ             | 11.06       | 8.25        | 6.78        | 52.49      | 60.98      | 65.94      |
+            +           +------------------+-------------+-------------+-------------+------------+------------+------------+
|            |           | Qronos           | 11.62       | 8.50        | 7.06        | 51.58      | 59.72      | 65.54      |
+            +           +------------------+-------------+-------------+-------------+------------+------------+------------+
|            |           | Learned Round    | 11.01       | 8.38        | 6.70        | 53.05      | 61.11      | 65.64      |
+            +-----------+------------------+-------------+-------------+-------------+------------+------------+------------+
|            | MagR      | RTN              | 18.88       | 12.00       | 8.94        | 46.03      | 48.84      | 57.59      |
+            +           +------------------+-------------+-------------+-------------+------------+------------+------------+
|            |           | GPTQ             | 14.44       | 9.94        | 7.88        | 49.36      | 53.12      | 62.19      |
+            +           +------------------+-------------+-------------+-------------+------------+------------+------------+
|            |           | Qronos           | 13.19       | 8.94        | 7.50        | 51.28      | 58.27      | 63.86      |
+            +           +------------------+-------------+-------------+-------------+------------+------------+------------+
|            |           | Learned Round    | 12.48       | 9.50        |             | 50.86      | 59.18      |            |
+            +-----------+------------------+-------------+-------------+-------------+------------+------------+------------+
|            | QuaRot    | RTN              | 15.62       | 12.38       | 8.50        | 48.36      | 54.34      | 62.64      |
+            +           +------------------+-------------+-------------+-------------+------------+------------+------------+
|            |           | GPTQ             | 12.19       | 9.06        | 7.38        | 51.10      | 58.52      | 64.59      |
+            +           +------------------+-------------+-------------+-------------+------------+------------+------------+
|            |           | Qronos           | 11.81       | 8.62        | 7.00        | 51.71      | 59.06      |            |
+            +           +------------------+-------------+-------------+-------------+------------+------------+------------+
|            |           | Learned Round    | 11.69       | 8.40        | 6.86        | 52.35      | 60.26      | 41.28      |
+            +-----------+------------------+-------------+-------------+-------------+------------+------------+------------+
|            | SpinQuant | RTN              | 12.00       | 8.75        | 7.16        | 51.92      | 59.35      | 66.01      |
+            +           +------------------+-------------+-------------+-------------+------------+------------+------------+
|            |           | GPTQ             | 12.38       | 9.62        | 8.12        | 51.06      | 58.37      | 62.93      |
+            +           +------------------+-------------+-------------+-------------+------------+------------+------------+
|            |           | Qronos           | 11.62       | 8.62        | 7.22        | 51.50      | 59.26      | 64.49      |
+            +           +------------------+-------------+-------------+-------------+------------+------------+------------+
|            |           | Learned Round    | 11.71       | 8.51        | 6.93        | 52.52      | 59.79      |            |
+------------+-----------+------------------+-------------+-------------+-------------+------------+------------+------------+

.. rubric:: References

.. [1] Nagel, M., Amjad, R. A., Van Baalen, M., Louizos, C., & Blankevoort, T. (2020, November). Up or down? adaptive rounding for post-training quantization. In International conference on machine learning (pp. 7197-7206). PMLR.
.. [2] Cheng, W., Zhang, W., Shen, H., Cai, Y., He, X., Kaokao, L., & Liu, Y. (2024, November). Optimize weight rounding via signed gradient descent for the quantization of llms. In Findings of the Association for Computational Linguistics: EMNLP 2024 (pp. 11332-11350).
.. [3] Frantar, E., Ashkboos, S., Hoefler, T., & Alistarh, D. (2022). Gptq: Accurate post-training quantization for generative pre-trained transformers. arXiv preprint arXiv:2210.17323.
.. [4] Zhang, S., Zhang, H., Colbert, I., & Saab, R. (2025). Qronos: Correcting the Past by Shaping the Future... in Post-Training Quantization. arXiv preprint arXiv:2505.11695.
.. [5] Ashkboos, S., Mohtashami, A., Croci, M. L., Li, B., Cameron, P., Jaggi, M., ... & Hensman, J. (2024). Quarot: Outlier-free 4-bit inference in rotated llms. Advances in Neural Information Processing Systems, 37, 100213-100240.
.. [6] Liu, Z., Zhao, C., Fedorov, I., Soran, B., Choudhary, D., Krishnamoorthi, R., ... & Blankevoort, T. (2024). Spinquant: Llm quantization with learned rotations. arXiv preprint arXiv:2405.16406.
.. [7] Zhang, A., Wang, N., Deng, Y., Li, X., Yang, Z., & Yin, P. (2024). Magr: Weight magnitude reduction for enhancing post-training quantization. Advances in neural information processing systems, 37, 85109-85130.
.. [8] Shao, W., Chen, M., Zhang, Z., Xu, P., Zhao, L., Li, Z., ... & Luo, P. (2023). Omniquant: Omnidirectionally calibrated quantization for large language models. arXiv preprint arXiv:2308.13137.
